# xNN
TODO: description


# REQUIREMENTS
Before running the programm, install **python=3.8** and the required libraries listed in **requirements.txt** file. A environment manager is suggested (i.e. conda)
~~~
# OPTIONAL: Environment creation (with conda).
conda create -n xnn_env python=3.8
# OPTIONAL: Environment activation (with conda).
conda activate xnn_env

# Dependencies installed.
pip install -r requirements.txt
~~~

Note that the code is optimized for:

- python=3.8
- tensorflow=2.9


# CONFIGURATION
The program receives its configuration from a file located in ./configuration/config.json. The parameters that need to be configured are:

- **pad_values**: The pad value to use when padding="same". It's dafult value is 0.0.
- **input_quantizer**: Quantization function applied to the incoming activations. It's default value is "ste_tern", but for the first layer, which is "null". More info: https://docs.larq.dev/larq/guides/key-concepts/#quantized-layers
- **depthwise_quantizer**: Quantization function applied to the depthwise kernel. It's default value is "ste_tern". Used in the following layers: QuantDepthwiseConv2D, QuantSeparableConv1D, QuantSeparableConv2D. 
- **pointwise_quantizer**: Quantization function applied to the pointwise kernel. It's default value is "ste_tern". Used in the following layers: QuantSeparableConv1D, QuantSeparableConv2D.
- **kernel_quantizer**: Quantization function applied to the kernel weights matrix of the layer. It's default value is "ste_tern". More info: https://docs.larq.dev/larq/guides/key-concepts/#quantized-layers

The available quantizers are: approx_sign, ste_heaviside, swish_sign, magnitude_aware_sign, ste_tern, dorefa_quantizer. They are instanced using their default values. For more information about them, refer to LARQ documentation (https://docs.larq.dev/larq/api/quantizers/)
https://docs.larq.dev/larq/guides/key-concepts/#quantized-layers
Notice here that **if all xxx_quantizer function are 'null'** the layer is equivalent to its corresponding full precision layer.

# MAPPING A DNN INTO A BNN
This command translate a DNN into a BNN. This means that all the layers with a translation supported by LARQ will be mapped. The rest of the layers (i.e. Flatten) will be exactly copied from the original.

To run the problem getting on one single instance, the user should be in the same directory as app.py. The command to invoke is:
~~~
python app.py dnn2bnn --fin *path_to_dnn_model* --fout *path_to_bnn_model*
~~~

Where the options are:

**--fin**: Path to the file containing the DNN model to be mapped.

**--fout**: Path to the file where the BNN model will be saved.

Example:
~~~
python app.py dnn2bnn --fin ./models_in/mnist_mlp_2h.h5 --fout ./models_out/bin_mnist_mlp_2h.h5
~~~

## USE CASE
Example of use case:
We want to predict hand-written numbers (dataset: MNIST). First we train a MLP of two layers with this dataset and mapped it to BNN in order to compare performances. One likely process could be:

1) To prepare the dataset and its labels for the training phase. Build a simple MLP of two layers, train it and save the model. Code available in dnn_examples/mlp.ipynb
2) To map the DNN and save the obtained BNN using the command "dnn2bnn".
3) To prepare the dataset and its labels in the same way we did it for the DNN. Load the resulting model of step 2) and trained it. Code available in dnn_examples/bmlp.ipynb

# TEST
From main folder invoke:
python -m unittest discover test

# TODO
https://docs.larq.dev/larq/tutorials/binarynet_cifar10/

# USE AS LIBRARY.
In root directory execute:
python setup.py bdist_wheel

pip install ./dist/dnn2bnn-0.1.0-py3-none-any.whl
pip install ./dist/dnn2bnn-0.1.0-py3-none-any.whl --force-reinstall