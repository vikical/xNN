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
- tensorflow=2.3
- cuda=10.1

To change this requirement to adjust the code to different environments, please, take into account the following installation guide: https://www.tensorflow.org/install/source#gpu.

# CONFIGURATION
The program receives its configuration from a file located in ./configuration/config.py. The parameters that need to be configured are:

**--param1**: description. Its default value is null.

**--param2**: description.

**--param3**: description.


# RUNNING ONE SINGLE INSTANCE
TODO: update
To run the problem getting on one single instance, the user should be in the same directory as app.py. The command to invoke is:
~~~
python app.py dnn2bnn --fin *path_to_dnn_model* --fout *path_to_bnn_model* --ftrain *path_to_training_dataset*
~~~

Where the options are:

**--fin**: description

**--fout**: description

**--ftrain**: description


Example:
~~~
python app.py dnn2xnn --din ...............
python app.py dnn2xnn --din ...............
~~~


# TEST
From main folder invoke:
python -m unittest discover test