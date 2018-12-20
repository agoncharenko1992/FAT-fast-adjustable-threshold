import os
import pickle
import json
from typing import Any

import tensorflow as tf


# tf.Graph related functions

def load_pb(filename) -> tf.GraphDef:
    """Loads graph definitions from the specified file.

    Only TensorFlow models saved as *.pb files are supported

    Parameters
    ----------
    filename: str
        A path to the file containing definitions of the graph

    Returns
    -------
    tf.GraphDef:
        A TensorFlow GraphDef object
    """
    graph_def = tf.GraphDef()
    with tf.gfile.GFile(filename, 'rb') as fid:
        graph_def.ParseFromString(fid.read())
    return graph_def


def graph_from_graph_def(graph_def):
    """Converts the provided GraphDef object into a TensorFlow Graph

    Parameters
    ----------
    graph_def: tf.GraphDef
        A GraphDf object to be converted

    Returns
    -------
    tf.Graph:
        A TensorFlow static graph
    """
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        tf.import_graph_def(graph_def, name='')
    return tf_graph


def save_pb(graph, target_path):
    """Saves a TensorFlow graph to a file.

    Parameters
    ----------
    graph: tf.Graph
        A model to save.
    target_path: str
        The name of the output file in which the provided graph will be saved.
    """
    _ = tf.train.write_graph(graph,
                             logdir=os.path.dirname(target_path),
                             name=os.path.basename(target_path),
                             as_text=False)


def save_event(model, save_dir):
    """Saves the data of the provided model as a TensorFlow event file.

    The resulting file can be used to visualize data of a TensorFlow Graph only.

    Parameters
    ----------
    model: tf.Graph or tf.GraphDef
    save_dir: str

    """
    if isinstance(model, tf.Graph):
        train_writer = tf.summary.FileWriter(save_dir)
        train_writer.add_graph(model)
    elif isinstance(model, tf.GraphDef):
        train_writer = tf.summary.FileWriter(save_dir)
        train_writer.add_graph(graph_from_graph_def(model))
    else:
        raise TypeError("'model' must be whether a graph or a graph_def object")


# Pickle files

def load_pickle(filename) -> Any:
    """Loads any given pickle file.

    Parameters
    ----------
    filename: str
        The path to a pickle file
    """
    return pickle.load(open(filename, 'rb'))


def save_pickle(data, filename):
    """Saves data to a pickle file.

    Parameters
    ----------
    data: Any
        Data to be saved
    filename: str
        A path to a file in which the provided data will be saved
    """
    pickle.dump(data, open(filename, 'wb'))
        

# Working with folders

def clear_dir(directory_path):
    """Removes the content of the specified folder.

    Parameters
    ----------
    directory_path:
        A path to a folder which content must be removed.

    """
    if not os.path.isdir(directory_path):
        raise NotADirectoryError("Specified path is not a folder")
    
    file_list = [os.path.join(directory_path, file_name) for file_name in os.listdir(directory_path)]
    
    for file_name in file_list:
        if os.path.isdir(file_name):
            raise RuntimeError("The specified folder contains another folder (recurcive clean is prohibited)")
    
    for file_name in file_list:
        os.remove(file_name)


def create_folder_from_path_if_necessary(path):
    """Creates all necessary folders for the specified path.

    Parameters
    ----------
    path: str
        A path from which folder names and structure will be extracted

    """
    realpath = os.path.realpath(path)
    dirname = os.path.dirname(realpath)
    
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def check_paths_exist(*paths):
    """Check if every specified path path among the specified exists.

    Parameters
    ----------
    paths: list of str
        A list of paths to check
    """
    for path in paths:
        realpath = os.path.realpath(path)
        if not os.path.exists(realpath):
            raise FileNotFoundError("Path '{}' doesn't exist".format(realpath))


# Other

def googlenet_preprocess(image):
    """Preprocess input data using GoogLeNet preprocess method.

    Parameters
    ----------
    image:
        An input image (or a tensor) that should be preprocessed
    """
    return image / 127.5 - 1.


def load_json(filename):
    """Loads data from a .json file

    Parameters
    ----------
    filename: str
        A path to the *.json file
    """

    if not os.path.exists(filename):
        raise FileNotFoundError("'{}' not found".format(filename))

    return json.load(open(filename, 'r'))
