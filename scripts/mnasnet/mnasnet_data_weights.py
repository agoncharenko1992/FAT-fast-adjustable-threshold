import tensorflow as tf
import numpy as np
from copy import deepcopy


def _retrieve_data_links(op_scope, bn_scope=None, bias_scope=None, base_scope="mnasnet"):
    data = {}
    
    if "depthwise" in op_scope:
        data['weights'] = base_scope + '/' + op_scope + '/depthwise_kernel:0'
    else:
        data['weights'] = base_scope + '/' + op_scope + '/tf_layer/kernel:0'
    
    if bn_scope is not None:
        data["gamma"] = base_scope + '/' + bn_scope + '/tf_layer/gamma:0'
        data["beta"] = base_scope + '/' + bn_scope + '/tf_layer/beta:0'
        data["mean"] = base_scope + '/' + bn_scope + '/tf_layer/moving_mean:0'
        data["variance"] = base_scope + '/' + bn_scope + '/tf_layer/moving_variance:0'
    
    if bias_scope is not None:
        data["bias"] = base_scope + '/' + bias_scope + '/tf_layer/bias:0'
    
    return data


def _gw(sess: tf.Session, op_scope, bn_scope=None, bias_scope=None, base_scope="mnasnet"):
    return sess.run(_retrieve_data_links(op_scope, bn_scope, bias_scope, base_scope))


def _get_cascade_cell_name(operation_names, cell_index):
    substr = "cell_" + str(cell_index) + '/'
    for op_name in operation_names:
        if substr in op_name:
            return op_name.split('/')[1]
    return None


def _fold_bn(weights, moving_mean, moving_variance, gamma, beta, eps=10**-3):
    scale_factor = gamma / np.sqrt(moving_variance + eps)
    bias = beta - moving_mean * scale_factor
    weights = np.multiply(weights, scale_factor)
    return weights, bias


def _fold_bn_dws(weights, moving_mean, moving_variance, gamma, beta, eps=10**-3):
    scale_factor = gamma / np.sqrt(moving_variance + eps)
    bias = beta - moving_mean * scale_factor
    weights = np.multiply(weights, np.expand_dims(scale_factor, 1))
    return weights, bias


def _fold_weights(weights_dict):
    folded_weights = dict()
    
    for ln, w_data in weights_dict.items():
        new_w_data = {}
        w = w_data["weights"]
        
        if "dws" in ln:
            w, b = _fold_bn_dws(w_data["weights"], 
                                w_data["mean"], 
                                w_data["variance"], 
                                w_data["gamma"], 
                                w_data["beta"])
            new_w_data["weights"] = w
            new_w_data["bias"] = b
        elif "fc" in ln:
            new_w_data = w_data
        else:
            w, b = _fold_bn(w_data["weights"], 
                            w_data["mean"], 
                            w_data["variance"], 
                            w_data["gamma"], 
                            w_data["beta"])
            new_w_data["weights"] = w
            new_w_data["bias"] = b
        
        folded_weights[ln] = deepcopy(new_w_data)
    
    return folded_weights


def get_weights_for_mnasnet(model: tf.Graph) -> dict:
    """Extracts weights dictionary from any MNasNet model hosted at
    **www.tensorflow.org/lite/models**

    Weights are preprocessed in order to eliminate operations, related to batch normalization.

    Parameters
    ----------
    model: tf.Graph
        A static graph, from which weights data must be extracted.

    Returns
    -------
    dict:
        A dictionary containing layers' weights data (including biases)
    """
    with tf.Session(graph=model) as sess:
    
        operation_names = [op.name for op in model.get_operations() if op.type=="Const"]

        weights = {}

        # stem cell
        weights["stem"] = _gw(sess, "stem/conv", "stem/bn")

        # 0-th cascade cell
        weights["lead_cell_0/dws"] = _gw(sess, "lead_cell_0/op_0/depthwise_0", "lead_cell_0/op_0/bn1_0")
        weights["lead_cell_0/project"] = _gw(sess, "lead_cell_0/op_0/project_0", "lead_cell_0/op_0/bn2_0")

        # last cascade cell
        weights["lead_cell_17"] = _gw(sess, "lead_cell_17/op_0/conv2d_0", "lead_cell_17/op_0/bn_0")

        # intermediate cascade cells
        for cell_index in range(1, 17):
            cell_scope = _get_cascade_cell_name(operation_names, cell_index)
            # expand -> dws -> project
            weights[cell_scope + "/expand"] = _gw(sess, cell_scope + "/op_0/expand_0", cell_scope + "/op_0/bn0_0")
            weights[cell_scope + "/dws"] = _gw(sess, cell_scope + "/op_0/depthwise_0", cell_scope + "/op_0/bn1_0")
            weights[cell_scope + "/project"] = _gw(sess, cell_scope + "/op_0/project_0", cell_scope + "/op_0/bn2_0")

        # output cell
        weights["output/fc"] = _gw(sess, "output/fc", bias_scope="output/fc")
    
    return _fold_weights(weights)
