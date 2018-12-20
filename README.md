# FAT (Fast Adjustable Thresholds) Arxiv: https://arxiv.org/abs/1812.07872

## Table of Content
 * [Requirements](#requirements)
 * [Train quantization thresholds for MNasNet](#train-quantization-thresholds-for-mnasnet)
 * [DataGenerator](#using-datagenerator)
 * [Training quantization thresholds](#training-quantization-thresholds)
 * [Trained Quantized MNasNet models](#trained-quantized-mnasnet-models)
 * [Authors](#authors)

## Requirements
The following libraries are required:
 * **numpy** - 1.14.4
 * **opencv-python** - 3.4.1.15
 * **tensorflow-gpu** - 1.8.0 
 * **tqdm** - 4.28.1
 
> If you want to use CPU instead of GPU replace **tensorflow-gpu** with **tensorflow**.

> Version of libraries you use may differ from the versions we specify in requirements.

## Train quantization thresholds for MNasNet

We provide support for the MNasNet models which are hosted at 
<www.tensorflow.org/lite/models>. Download any MNasNet model you are intrested in and extract a __*.pb__ file from it.

Quantization of MNasNet models is performed in a TFLite manner, so the resulted structure
of quantized models differs from the original.

#### Extract weights from an existing model

All MNasNet models hosted by TensorFlow have the same structure and differ in size of 
kernels of convolution layers. So the information about the model's weights is enough
to restore the model itself.

To extract weights from the downloaded __*.pb__ file use the following command:

```
$ python prepare_weights.py /path/to/the/mnasnet/model.pb
```
or if you use GPUs and want to specify which one will be used for calculations
```
$ CUDA_VISIBLE_DEVICES=0 python prepare_weights.py /path/to/the/mnasnet/model.pb
```

It will create a __*.pickle__ file and place it in the same folder as the model:

> */path/to/the/mnasnet/model_weights.pickle*

You also can use the notebook [Prepare MNasNet weights.ipynb](Prepare MNasNet weights.ipynb)

#### Quantize and train the quantization thresholds of MNasNet models

We provide the notebook [Train Thresholds.ipynb](Train Thresholds.ipynb) to quantize 
MNasNet models and to train its quantization thresholds.

It utilizes the following classes:
 * [DataGenerator](#using-datagenerator)
 * [Trainer](#training-quantization-thresholds)
 
You need to follow several steps:

> You can find additional comments in the notebook
 
 1) Specify the path to the __*.pickle__ file containing weights of the model you want to build.
 2) Specify the base output folder, where the checkpoints and model's adjusted threshold will be stored. 
    The hierarchy of the output data is following:
    ```
    .specified_output_folder/
      |
      o -- model_name/
            |
            | -- ckpt/
            |     L ...
            |
            | -- best_ckpt/
            |     L ...
            |
            | -- model_thresholds.pickle    (will be created during training)
            o -- model_fakequant.pb    (will be created during training)
    ```
 3) Setup data generators
    1. Specify the paths to the base folders containing training and validation images
    2. Specify the paths to the lists containing the pairs **(image name, image label)**.
       The format of the lists must be following:
       ```
       relative/path/to/image_1.JPEG label_1
       relative/path/to/image_2.JPEG label_2
       ...
       ```
       The paths to the images must be relative to the corresponding images folders.
 
 4) Set up calibration parameters (the number of calibration batches and 
    the number of images per batch)
 
 5) Adjust parameters of the training. You need to modify 
    [settings_config/train.json](settings_config/train.json).
    
    > In the notebook `Train Thresholds.ipynb` we ignore such parameters as 
    `save_dir` and `best_ckpt_dir` and override them with paths created exclusively 
    for the MNasNet models.
    
 6) Run the model training.
    
    > It can take a while, depending on the number of epochs and trainable images 
 
 7) Use adjusted thresholds values to build a MNasNet model with fake quant nodes.
    The output model is compatible with TFLite.
    
    > The output model is saved as a __*.pb__ file. You need to use external tools 
    (like *tensorflow toco*) to convert it to TFLite format.

## Using DataGenerator
**DataGenerator** is an instrument that allows you to iterate over the images in a simple way.

During initialization **DataGenerator** expects two paths:
 * path to the folder containing images or other folders with images
 * path to the list of images with corresponding labels in the following format:
   ```
   relative/path/to/image_1.JPEG label_1
   relative/path/to/image_2.JPEG label_2
   ...
   ```
   
   Path to each image must be relative to the specified folder.

**DataGenerator** has a public method `generate_batches(...)` which creates a generator that 
iterates over the image dataset. This generator yields pairs 
(**batch of images**, **batch of labels**) so it can be used for calibration and validation as well.

See the **DataGenerator** [source code](scripts/data/data_generator.py) for more details.

## Training quantization thresholds
Class **Trainer** provides basic functionality to train the model by minimizing the difference 
between the output of the trained model and the output of the reference model.

Both the reference model and the trainable one must be in the same graph and have the same 
input node.

The following parameters are the most important for the training process and must be 
adjusted for each task individually:

 * `learning_rate`
 * `learning_rate_decay`
 * `batch_size`
 * `epochs`
 * `reinit_adam_after_n_batches`

We prefer to store these parameters in the external file 
[settings_config/train.json](settings_config/train.json). However, you can define these
parameters wherever you want, just make sure you feed them to the \_\_init\_\_ method
of the **Trainer**

After instantiating the **Trainer** class you will be able to invoke two main methods: 
`train` and `validate`. Each accept the session as an input argument and expect 
all variables to be initialized.

See the **Trainer** [source code](scripts/trainer/trainer.py) for more details.

## Trained Quantized MNasNet models

We provide the quantized MNasNet models built with trained quantization thresholds.
Names of the input and the output of the models are **input_node** and **output_node**
correspondingly.

| *.pb-file with fake quant nodes | TFLite model | TFLite model accuracy (Top 1, %) |
|:-------------------------------:|:------------:|:--------------------------------:|
| [mnasnet_0.5_224_quant.pb][m1_pb] | [mnasnet_0.5_224_quant.tflite][m1_tflite] | 66.6 |
| [mnasnet_0.75_224_quant.pb][m2_pb] | [mnasnet_0.75_224_quant.tflite][m2_tflite] | 70.11 |
| [mnasnet_1.0_128_quant.pb][m3_pb] | [mnasnet_1.0_128_quant.tflite][m3_tflite] | 66.76 |
| [mnasnet_1.0_224_quant.pb][m4_pb] | [mnasnet_1.0_224_quant.tflite][m4_tflite] | 72.45 |
| [mnasnet_1.3_224_quant.pb][m5_pb] | [mnasnet_1.3_224_quant.tflite][m5_tflite] | 74.74 |

[m1_pb]: https://drive.google.com/uc?export=download&id=1HZJ6Pazsd2DuvXYwVQKxdU4FOs5h68hm
[m1_tflite]: https://drive.google.com/uc?export=download&id=1p4t384QpX5UIRpM-787FgOKL0yxo4xZ-

[m2_pb]: https://drive.google.com/uc?export=download&id=1KefZ7Vj5rF_CNl6MTiZ5dhUCO4a58Vwb
[m2_tflite]: https://drive.google.com/uc?export=download&id=1pBwsN_b1itnxen62tdX8wfhn4wpuLhm-

[m3_pb]: https://drive.google.com/uc?export=download&id=11eJ_Zx762z4Iu3kBFa0bAPW-PLzhKRY9
[m3_tflite]: https://drive.google.com/uc?export=download&id=19ScyBJyQAaEBGdgQ8pHCxBz--SyuNPUj

[m4_pb]: https://drive.google.com/uc?export=download&id=1kMVRxYOcnQ_kv5qz5YMB2ZfWHztKcbgO
[m4_tflite]: https://drive.google.com/uc?export=download&id=1032Wv2itCqjxNwyscD-x6MRPjavR8Amg

[m5_pb]: https://drive.google.com/uc?export=download&id=19srh4rqAsyBp3AC8v7ammL69y9TwjOKF
[m5_tflite]: https://drive.google.com/uc?export=download&id=1gQP6iGRdzMmvczEfDcuJyWsHlqpXQA_B

## Authors

 * Goncharenko Alexander
 * Denisov Andrey
