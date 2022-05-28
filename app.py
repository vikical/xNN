from genericpath import exists
import click, os, logging, sys, larq
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils import layer_utils
import json


def set_logging(log:str):
    log_level=logging.getLevelName(log)
    logging.basicConfig(stream=sys.stderr, level=log_level)

@click.group()
def xnn():
    pass

@xnn.command()
def man():
    '''
    Shows man
    '''
    click.echo("""To invoke the app the following commands can be used (without --):
           aaa: description
           bbb: description
           man: show the manual""")


@xnn.command()
@click.option("--din", type=click.Path(exists=True), help="Path to the directory where files with distances between nodes and for vehicles are located. This files should follow reg. exp. dist_NUMBER.txt and veh_dist_NUMBER.txt")
@click.option("--metaheu", default="" , help="Comma-separed list of metaheuristics which are going to be compare. Options: ls (local search); ts (tabu search); vnd; bvns")
def testbench(din,metaheu,niter,search,init,memory,times4ave,log,maxs):
    set_logging(log=log)

    logging.info("STARTING TEST BENCH...")

    print("TEST BENCH DONE AND SAVED!!!")



@xnn.command()
@click.option("--din", type=click.Path(exists=True), help="Path to the directory where files with distances between nodes and for vehicles are located. This files should follow reg. exp. dist_NUMBER.txt and veh_dist_NUMBER.txt")
@click.option("--problem", default="0", help="Number of the problem instance to be solved, i.e. the NUMBER in the dist_NUMBER.txt files")
def solve(din,problem,metaheu,niter,search,init,memory,log,maxs):
    set_logging(log=log)

    print("\n\n ---------FINISH---------")
   

@xnn.command()
@click.option("--din", type=click.Path(exists=True), help="Path to the directory where files with distances between nodes are located. This files should follow reg. exp. dist_NUMBER.txt")
def dnn2bnn(din):
    '''
    Translate a given trained dnn to its corresponding bnn.
    '''   

    model = tf.keras.Sequential([
        hub.KerasLayer("https://tfhub.dev/adityakane2001/regnety200mf_classification/1")
    ])
    model.build(input_shape=(None,224,224,3))
    my_summary=model.summary()

    print("DNN layers:")
    for layer in model.layers:
        print(layer)

    x = tf.keras.Input(shape=(28, 28, 1))  
    y = tf.keras.layers.Conv2D(32, 3, strides=2, padding="same")(x)
    model = tf.keras.Model(inputs=x, outputs=y)
    model.build(input_shape=(28, 28, 1))
    model.summary()


    #Build BNN
    x = tf.keras.Input(shape=(28, 28, 1))
    y = tf.keras.layers.Flatten()(x)
    y = larq.layers.QuantDense(
        512, kernel_quantizer="ste_sign", kernel_constraint="weight_clip")(y)
    y = larq.layers.QuantDense(
        10,
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
        activation="softmax")(y)
    model = tf.keras.Model(inputs=x, outputs=y)

    # print("BNN layers:")
    # for layer in model.layers:
    #     print(layer)

if __name__ == '__main__':
    xnn()