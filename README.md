# xNN
This is a library to translate a full-precision DNN into a BNN. The library also contains a fidelity metric to evaluate to which extent the translation mimic the original model. In the directory dnn_examples there are 4 python files that can serve as examples of the library use.

Additionally there is an application (app.py) which takes as input a DNN and return a BNN ready to be trained.


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

- **kernel_constraint**: Function to be applied to latent weights. For example "weight_clip" that restricts the weights between [-1,1].
- **activation_relu**: Maps the activation function "relu" to another function. It is also allowed "null".
- **activation_sigmoid**: Maps the activation function "sigmoid" to another function. It is also allowed "null".
- **activation_softmax** : Maps the activation function "softmax" to another function. It is also allowed "null".

Other activation functions can be customized, just by adding a new parameter following the same pattern "activation_".

# INSTALLING THE LIBRARY
In root directory execute:
~~~
python setup.py bdist_wheel

pip install ./dist/dnn2bnn-0.1.0-py3-none-any.whl --force-reinstall
~~~

# USING dnn2bnn AS LIBRARY
Import the required resources from "dnn2bnn":
~~~
from dnn2bnn.metrics.fidelity import Fidelity
from dnn2bnn.models.model_manager import ModelManager
~~~
Load LARQ configuration.
~~~
larq_configuration={}
with open("../whatever_path/whatever_config.json") as json_data_file:
    larq_configuration = json.load(json_data_file)
~~~
Define a DNN model and compile it. It can be trained, but it is not mandatory.
~~~
inputs=...
model_dnn = Model(inputs=inputs, outputs=layer)
model_dnn.compile(optimizer=...,loss=..., metrics=[...])
~~~
Translate the full-precision DNN model into a BNN.
~~~
mm=ModelManager(original_model=model_dnn,larq_configuration=larq_configuration)
model_bin=mm.create_larq_model()    
model_bin.fit(x_train, y_train, epochs=epoch_size, batch_size=batch_size)  
~~~
Evaluate the fidelity.
~~~
fidelity=Fidelity(original=model_dnn, surrogate=model_bin,x=x_test)    
fidelity_value=fidelity.accuracy()
~~~

# app.py: MAPPING A DNN INTO A BNN
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

# TEST
From main folder invoke:
python -m unittest discover test


