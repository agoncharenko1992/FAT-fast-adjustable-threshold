import tensorflow as tf
import numpy as np

from .mnasnet_model_float import MNasNetModelFloat
from .mnasnet_model_float import MNasNetInitError


def _check_init_parameters(thresholds):
    if not isinstance(thresholds, dict):
        raise MNasNetInitError("Precalculated thresholds must be presented via a dictionary")

    if not all([isinstance(th_name, str) for th_name in thresholds.keys()]):
        raise MNasNetInitError("All names of thresholds must be strings")

    if not all([isinstance(th_data, dict) for th_data in thresholds.values()]):
        raise MNasNetInitError("Thresholds must be packed in pairs into dictionaries")

    if not all([("min" in th_data and "max" in th_data) for th_data in thresholds.values()]):
        raise MNasNetInitError("Each reference node must have corresponding minimal and maximal thresholds")


def _get_name_scope():
    return tf.get_default_graph().get_name_scope()


class MNasNetModelFakeQuant(MNasNetModelFloat):
    """Creates MNasNet model based on the specified weights with fake quant nodes.

    The resulting model is compatible with TFLite.

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

    def __init__(self, input_node, weights, thresholds, output_node_name="output_node"):
        _check_init_parameters(thresholds)
        self._initial_thresholds = thresholds
        super().__init__(input_node, weights, output_node_name)

    def _get_thresholds(self, thresholds_name):
        return self._initial_thresholds[thresholds_name]["min"], self._initial_thresholds[thresholds_name]["max"]

    def _create_weights_node(self, weights_data):
        weights_name_scope = _get_name_scope() + "/weights"
        
        w_min, w_max = self._get_thresholds(weights_name_scope)
        
        weights_node = tf.constant(weights_data, tf.float32, name="weights")
        self._add_reference_node(weights_node)

        quantized_weights = tf.fake_quant_with_min_max_args(weights_node,
                                                            w_min,
                                                            w_max,
                                                            name="quantized_weights")
        return quantized_weights

    def _cell_output(self, net, output_type=None):

        output_name_scope = _get_name_scope() + "/output"

        if output_type == "fixed":
            i_min, i_max = -1, 1
        else:
            i_min, i_max = self._get_thresholds(output_name_scope)

        net = tf.fake_quant_with_min_max_args(net, i_min, i_max, name="output")

        self._add_reference_node(net)

        return net
