import tensorflow as tf

from .mnasnet_model_fake_quant import MNasNetModelFakeQuant


def _rounding_fn(node, name):
    return tf.round(node, name)


def _get_name_scope():
    return tf.get_default_graph().get_name_scope()


def _fixed_quant_signed(input_node, min_th, max_th, bits=8, name="fixed_signed_data"):
    th_width = max_th - min_th

    with tf.name_scope(name):
        min_th_node = tf.constant(min_th, tf.float32, name="min_th")
        th_width_node = tf.constant(th_width, tf.float32, name="th_width")

        # scale
        with tf.name_scope("scale"):
            q_range = 2. ** bits - 1.

            eps = tf.constant(10. ** -7, dtype=tf.float32, name='eps')
            scale_node = tf.div(q_range, th_width_node + eps, "new_input_scale")

        with tf.name_scope("quantized_bias"):
            quant_bias = tf.multiply(min_th_node, scale_node, name="scaling")
            quant_bias = _rounding_fn(quant_bias, name="rounding")

        with tf.name_scope("discrete_input_data"):
            net = tf.multiply(input_node, scale_node, name="scaling")
            net = _rounding_fn(net, name='rounding')
            net = tf.clip_by_value(net - quant_bias, clip_value_min=0, clip_value_max=2 ** bits - 1) + quant_bias

            descrete_input_data = tf.div(net, scale_node, name="discrete_data")

    return descrete_input_data


class MNasNetModelAdjustable(MNasNetModelFakeQuant):
    """Creates MNasNet model based on the specified weights with adjustable fake quantization nodes.

    The resulting model is NOT compatible with TFLite and should be used for training only.

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
    adjusted_thresholds: dict
        A dictionary containing the values of the adjusted quantization thresholds.
        (Initially, these values, and the thresholds used for initialization are the same.
        Training must be conducted first.)
    variables: list
        A list of variables associated with the model
    initializer: list
        A list of initializers for variables, associated with the model
    """

    def __init__(self, input_node, weights, thresholds, output_node_name="output_node"):
        self._adjusted_thresholds = dict()
        self._ths_vars = list()
        super().__init__(input_node, weights, thresholds, output_node_name)
    
    def _add_thresholds(self, name, min_th, max_th):
        self._adjusted_thresholds[name] = {"min": min_th, "max": max_th}

    def _add_th_var(self, th_variable):
        self._ths_vars.append(th_variable)

    @property
    def adjusted_thresholds(self):
        return {th_name: dict(th_data) for th_name, th_data in self._adjusted_thresholds.items()}

    @property
    def variables(self):
        return list(self._ths_vars)

    @property
    def initializer(self):
        return list(var.initializer for var in self._ths_vars)

    def _adjustable_quant_signed(self, input_node, min_th, max_th, out_name, bits=8, name="adjust_signed_data"):

        th_width = max_th - min_th

        with tf.name_scope(name):
            with tf.name_scope("thresholds"):
                min_th_node = tf.constant(min_th, tf.float32, name="min_th")
                th_width_node = tf.constant(th_width, tf.float32, name="th_width")

                alpha = tf.Variable(1.0, dtype=tf.float32, name="th_width_scale")
                beta = tf.Variable(0.0, dtype=tf.float32, name="th_shift_percent")

                alpha_constrained = tf.clip_by_value(alpha, 0.5, 1.3, name="th_width_scale_constrain")
                beta_constrained = tf.clip_by_value(beta, -0.2, 0.4, name="th_shift_percent_constrain")

                adjusted_th_width = tf.multiply(th_width_node, alpha_constrained, name="adjusted_width")
                shift_node = tf.multiply(th_width_node, beta_constrained, name="min_th_shift")

                adjusted_min_th = tf.add(min_th_node, shift_node, name="adjusted_min_th")

                adjusted_max_th = tf.add(adjusted_min_th, adjusted_th_width, name="adjusted_max_th")

            # scale
            with tf.name_scope("scale"):
                q_range = 2. ** bits - 1.

                eps = tf.constant(10. ** -7, dtype=tf.float32, name='eps')
                scale_node = tf.div(q_range, adjusted_th_width + eps, "new_input_scale")

            with tf.name_scope("quantized_bias"):
                quant_bias = tf.multiply(adjusted_min_th, scale_node, name="scaling")
                quant_bias = _rounding_fn(quant_bias, name="rounding")

            with tf.name_scope("discrete_input_data"):
                net = tf.multiply(input_node, scale_node, name="scaling")
                net = _rounding_fn(net, name='rounding')
                net = tf.clip_by_value(net - quant_bias, clip_value_min=0, clip_value_max=2 ** bits - 1) + quant_bias

                descrete_input_data = tf.div(net, scale_node, name="discrete_data")

        self._add_thresholds(out_name, adjusted_min_th, adjusted_max_th)
        self._add_th_var(alpha)
        self._add_th_var(beta)

        return descrete_input_data

    def _adjustable_quant_unsigned(self, input_node, max_th, out_name, bits=8, name="adjust_unsigned_data"):

        th_width = max_th

        with tf.name_scope(name):
            with tf.name_scope("thresholds"):
                min_th_node = tf.constant(0, tf.float32, name="min_th")
                th_width_node = tf.constant(th_width, tf.float32, name="th_width")

                alpha = tf.Variable(1.0, dtype=tf.float32, name="th_width_scale")

                alpha_constrained = tf.clip_by_value(alpha, 0.5, 1.3, name="th_width_scale_constrain")

                adjusted_th_width = tf.multiply(th_width_node, alpha_constrained, name="adjusted_width")

            # scale
            with tf.name_scope("scale"):
                q_range = 2. ** bits - 1.
                eps = tf.constant(10. ** -7, dtype=tf.float32, name='eps')
                scale_node = tf.div(q_range, adjusted_th_width + eps, "new_input_scale")

            with tf.name_scope("discrete_input_data"):
                net = tf.multiply(input_node, scale_node, name="scaling")
                net = _rounding_fn(net, name='rounding')
                net = tf.clip_by_value(net, clip_value_min=0, clip_value_max=2 ** bits - 1)
                descrete_input_data = tf.div(net, scale_node, name="discrete_data")

        self._add_thresholds(out_name, min_th_node, adjusted_th_width)
        self._add_th_var(alpha)

        return descrete_input_data

    def _create_weights_node(self, weights_data):
        weights_name_scope = _get_name_scope() + "/weights"

        w_min, w_max = self._get_thresholds(weights_name_scope)

        weights_node = tf.constant(weights_data, tf.float32, name="weights")
        self._add_reference_node(weights_node)

        quantized_weights = self._adjustable_quant_signed(weights_node,
                                                          w_min,
                                                          w_max,
                                                          weights_name_scope,
                                                          name="quantized_weights")

        return quantized_weights

    def _cell_output(self, net, output_type=None):

        if output_type == "fixed":
            net = _fixed_quant_signed(net, -1, 1, name="fixed_quantized_input")
        else:
            output_name_scope = _get_name_scope() + "/output"
            i_min, i_max = self._get_thresholds(output_name_scope)

            if i_min == 0:
                net = self._adjustable_quant_unsigned(net, i_max, output_name_scope, name="quantized_input")
            else:
                net = self._adjustable_quant_signed(net, i_min, i_max, output_name_scope, name="quantized_input")

        net = tf.identity(net, name="output")

        self._add_reference_node(net)

        return net
