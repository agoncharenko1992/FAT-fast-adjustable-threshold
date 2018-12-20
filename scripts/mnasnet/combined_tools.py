import uuid
import os

import tensorflow as tf

from scripts.mnasnet.mnasnet_model_float import MNasNetModelFloat
from scripts.mnasnet.mnasnet_model_adjutable import MNasNetModelAdjustable
from scripts.mnasnet.mnasnet_model_fake_quant import MNasNetModelFakeQuant


def create_input_node(input_shape):
    with tf.Graph().as_default():
        return tf.placeholder(tf.float32, shape=input_shape, name="input_node")


def create_float_model(input_node, weights, output_node_name="output_node"):
    with input_node.graph.as_default():
        return MNasNetModelFloat(input_node, weights, output_node_name=output_node_name)


def create_adjustable_model(input_node, weights, thresholds):

    with input_node.graph.as_default():
        def _clip_grad_op(op, grad):
            x = op.inputs[0]
            x_min = op.inputs[1]
            x_max = op.inputs[2]
            cond = tf.logical_or(tf.less(x, x_min), tf.greater(x, x_max))
            return_grad = tf.where(cond, tf.zeros_like(grad, name="zero_grad"), grad)
            return return_grad, tf.constant(0, name="constant_min_grad"), tf.constant(0, name="constant_max_grad")

        # Register the gradient with a unique id
        grad_name = "MyClipGrad_" + str(uuid.uuid4())
        tf.RegisterGradient(grad_name)(_clip_grad_op)

        with input_node.graph.gradient_override_map({"Round": "Identity", "ClipByValue": grad_name}):
            mnasnet_model_quantized = MNasNetModelAdjustable(input_node, weights, thresholds)

        with tf.name_scope("float_model"):
            mnasnet_model_float = MNasNetModelFloat(input_node, weights)

    return mnasnet_model_float, mnasnet_model_quantized


def create_fakequant_model(input_node, weights, thresholds):
    with input_node.graph.as_default():
        return MNasNetModelFakeQuant(input_node, weights, thresholds)


def prepare_mnasnet_environment(pickle_path, output_dir, input_size=None, suffix="_weights.pickle"):
    if not isinstance(pickle_path, str):
        raise TypeError("Specified file name must be a string")

    if not isinstance(output_dir, str):
        raise TypeError("Specified name of the output directory must be a string")

    pickle_path = os.path.realpath(pickle_path)

    if not os.path.exists(pickle_path):
        raise FileNotFoundError("File '{}' not found".format(pickle_path))

    model_base_name = os.path.basename(os.path.realpath(pickle_path)).replace(suffix, "")

    model_output_dir = os.path.join(output_dir, model_base_name)
    checkpoint_folder = os.path.join(model_output_dir, "ckpt")
    best_checkpoint_folder = os.path.join(model_output_dir, "best_ckpt")

    thresholds_path = os.path.join(model_output_dir, model_base_name + "_thresholds.pickle")
    fakequant_output_path = os.path.join(model_output_dir, model_base_name + "_fakequant.pb")

    if not os.path.exists(checkpoint_folder):
        os.makedirs(checkpoint_folder)

    if not os.path.exists(best_checkpoint_folder):
        os.makedirs(best_checkpoint_folder)

    if input_size is not None:
        img_size = int(input_size)
    else:
        try:
            img_size = int(model_base_name.split('_')[-1])
        except (TypeError, ValueError):
            print("Unable to retrieve the iput size from the model name. 224x224 will be used")
            img_size = 224

    input_shape = (None, img_size, img_size, 3)

    print("Model: '{}'".format(model_base_name))
    print("INPUT_SHAPE:", input_shape)

    return input_shape, checkpoint_folder, best_checkpoint_folder, thresholds_path, fakequant_output_path
