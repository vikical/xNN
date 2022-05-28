import json

# Load configuration.
global CONFIG

CONFIG={"pad_values":0.0,
"input_quantizer":None,
"depthwise_quantizer":None,
"pointwise_quantizer":None,
"kernel_quantizer":None}

with open("configuration/config.json") as json_data_file:
    CONFIG = json.load(json_data_file)