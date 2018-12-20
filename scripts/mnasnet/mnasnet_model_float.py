import types
import tensorflow as tf


_lead_cells = (1, 4, 7, 10, 12, 16)  # except 0-th and 17-th
_lead_strides = (2, 2, 2, 1, 2, 1)


# MNasNet related exceptions
class MNasNetWeightsShapeError(Exception):
    pass


class MNasNetWeightsKeyError(Exception):
    pass


class MNasNetBuildError(Exception):
    pass


class MNasNetInitError(Exception):
    pass


class MNasNetModelFloat(object):
    """Creates MNasNet model based on the specified weights.

    Weights must be prepared, so that all batch normalization operations are fused with
    corresponding convolution operations.

    Properties
    ----------
    graph: tf.Graph
        A TensorFlow graph that hosts the model data
    input_node: tf.Tensor
        The input node of the MNasNet model
    output_node: tf.Tensor
        The output node of the MNasNet model
    reference_nodes: dict
        A dictionary containing all tensors which output data is necessary for
        calculating the quantization thresholds
    """

    def __init__(self, input_node, weights, output_node_name="output_node"):
        """

        Parameters
        ----------
        input_node
        weights
        output_node_name
        """
        self._input_node = input_node
        self._graph = tf.get_default_graph()

        self._weights = weights

        self._output_node_name = output_node_name

        self._reference_nodes = dict()

        self._output_node = self._create_model()

    @property
    def graph(self):
        return self._graph

    @property
    def output_node(self):
        return self._output_node

    @property
    def input_node(self):
        return self._input_node

    @property
    def reference_nodes(self):
        return dict(self._reference_nodes)

    def _add_reference_node(self, node: tf.Tensor):
        ref_node_name = node.name.split(":")[0]
        self._reference_nodes[ref_node_name] = node

    def _get_layer_weights(self, layer_name):
        if layer_name not in self._weights:
            raise MNasNetWeightsKeyError("Weights for the layer '{}' are not provided".format(layer_name))

        return self._weights[layer_name]

    def _cell_output(self, net, output_type="identity"):
        if output_type not in ["identity", "relu", "fixed"]:
            raise MNasNetBuildError("Unsupported type of cell output was specified: " + str(output_type))

        if output_type == "relu":
            net = tf.nn.relu(net, name="Relu")

        net = tf.identity(net, "output")
        self._add_reference_node(net)

        return net

    def _create_weights_node(self, weights_data):
        net =  tf.constant(weights_data, tf.float32, name="weights")
        self._add_reference_node(net)
        return net

    def _lead_cell(self, net, cell_scope, strides=2):

        with tf.name_scope(cell_scope):
            with tf.name_scope("expand"):
                net = self._build_conv(net, self._get_layer_weights(cell_scope + "/expand"), 1)
                net = self._cell_output(net, "relu")
                
            with tf.name_scope('dws'):
                net = self._build_dws(net, self._get_layer_weights(cell_scope + "/dws"), strides)
                net = self._cell_output(net, "relu")
            
            with tf.name_scope("project"):
                net = self._build_conv(net, self._get_layer_weights(cell_scope + "/project"), 1)
                net = self._cell_output(net, "identity")
        
        return net

    def _cell_with_skip_connection(self, net, cell_scope):

        input_node = net

        with tf.name_scope(cell_scope):
            with tf.name_scope("expand"):
                net = self._build_conv(net, self._get_layer_weights(cell_scope + "/expand"), 1)
                net = self._cell_output(net, "relu")

            with tf.name_scope('dws'):
                net = self._build_dws(net, self._get_layer_weights(cell_scope + "/dws"), 1)
                net = self._cell_output(net, "relu")

            with tf.name_scope("project"):
                net = self._build_conv(net, self._get_layer_weights(cell_scope + "/project"), 1)
                net = self._cell_output(net, "identity")

            with tf.name_scope("add"):
                net = tf.add(net, input_node)
                net = self._cell_output(net, "identity")

        return net
    
    def _build_dws(self, input_node, weights_and_bias, strides):
        weights_node = self._create_weights_node(weights_and_bias["weights"])
        
        op_output = tf.nn.depthwise_conv2d(input_node, 
                                           weights_node, 
                                           (1, strides, strides, 1), 
                                           padding="SAME")
        
        if "bias" in weights_and_bias:
            bias_node = tf.constant(weights_and_bias["bias"], tf.float32, name="bias")
            op_output = tf.nn.bias_add(op_output, bias_node)
        
        return op_output
    
    def _build_conv(self, input_node, weights_and_bias, strides):
        weights_node = self._create_weights_node(weights_and_bias["weights"])
        
        op_output = tf.nn.conv2d(input_node, 
                                 weights_node, 
                                 (1, strides, strides, 1), 
                                 padding="SAME")
        
        if "bias" in weights_and_bias:
            bias_node = tf.constant(weights_and_bias["bias"], tf.float32, name="bias")
            op_output = tf.nn.bias_add(op_output, bias_node)
        
        return op_output
    
    def _build_fc(self, input_node, weights_and_bias):
        weights_node = self._create_weights_node(weights_and_bias["weights"])
        
        op_output = tf.matmul(input_node, weights_node)
        
        if "bias" in weights_and_bias:
            bias_node = tf.constant(weights_and_bias["bias"], tf.float32, name="bias")
            op_output = tf.nn.bias_add(op_output, bias_node)
        
        return op_output

    def _build(self):

        with tf.name_scope("input_data"):
            net = self._cell_output(self._input_node, "fixed")

        # stem
        with tf.name_scope("stem"):
            net = self._build_conv(net, self._get_layer_weights("stem"), 2)
            net = self._cell_output(net, "identity")

        # lead_cell_0
        with tf.name_scope('lead_cell_0'):
            with tf.name_scope('dws'):
                net = self._build_dws(net, self._get_layer_weights("lead_cell_0/dws"), 1)
                net = self._cell_output(net, "relu")
            
            with tf.name_scope("project"):
                net = self._build_conv(net, self._get_layer_weights("lead_cell_0/project"), 1)
                net = self._cell_output(net, "identity")

        # series of cells with the similar structure
        for i in range(1, 17):

            if i in _lead_cells:
                cell_scope = "lead_cell_" + str(i)
                dws_stride = _lead_strides[_lead_cells.index(i)]
                net = self._lead_cell(net, cell_scope, dws_stride)

            else:
                cell_scope = "cell_" + str(i)
                net = self._cell_with_skip_connection(net, cell_scope)
        
        # lead_cell_17
        with tf.name_scope("lead_cell_17"):
            net = self._build_conv(net, self._get_layer_weights("lead_cell_17"), 1)
            net = self._cell_output(net, "relu")
        
        # output
        with tf.name_scope("output"):
            net = tf.reduce_mean(net, axis=(1, 2))
            with tf.name_scope("fc"):
                net = self._build_fc(net, self._get_layer_weights("output/fc"))
                net = self._cell_output(net, "identity")

        return tf.identity(net, name=self._output_node_name)

    def _create_model(self):
        try:
            output_node = self._build()
        except tf.errors.InvalidArgumentError as e:
            raise MNasNetWeightsShapeError("Specified weights for the model are inconsistent. " + e.message)

        return output_node
