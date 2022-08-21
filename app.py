from genericpath import exists
import click, logging, sys
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.python.keras.utils import layer_utils
import json

from dnn2bnn.models.model_manager import ModelManager

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
@click.option("--fin", type=click.File(mode="r"), help="File containing the model to be binarized")
@click.option("--fout", type=click.File(mode="w"), help="File containing the BNN model")
def dnn2bnn(fin, fout):
    '''
    Translate a given trained dnn to its corresponding bnn.
    '''   
    # Load configuration.
    larq_configuration={
        "reset_weights": False,
        "pad_values":0.0,
        "input_quantizer":None,
        "depthwise_quantizer":None,
        "pointwise_quantizer":None,
        "kernel_quantizer":None}

    with open("./configuration/config.json") as json_data_file:
        larq_configuration = json.load(json_data_file)


    original_model=tf.keras.models.load_model(filepath=fin.name)
    mm=ModelManager(original_model=original_model,larq_configuration=larq_configuration)
    larq_model=mm.create_larq_model()
    larq_model.save(fout.name)

    print("ORIGINAL MODEL")
    original_model.summary()
    print("BINARIZED MODEL")
    larq_model.summary()

if __name__ == '__main__':
    xnn()