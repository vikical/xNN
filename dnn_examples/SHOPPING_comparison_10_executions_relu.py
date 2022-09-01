import numpy as np
import tensorflow as tf
import pandas
import json

from dnn2bnn.metrics.fidelity import Fidelity
from dnn2bnn.models.model_manager import ModelManager

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Input, Conv1D, MaxPool1D, BatchNormalization

#Init global configuration.
batch_size=64 #64
epoch_size= 100
num_iter=5

#Load LARQ configuration.
larq_configuration={}

with open("../configuration/config_relu.json") as json_data_file:
    larq_configuration = json.load(json_data_file)

print(larq_configuration)

#Load data.
df=pandas.read_csv('online_shoppers_intention_preprocess.csv')

#Split training and test.
num_samples=len(df)
boundary_index=round(num_samples*0.85)

df_train=df[0:boundary_index]
df_test=df[boundary_index:]

print("Num. samples:"+str(num_samples)+ ". Training samples: " + str(len(df_train))+". Test samples: "+str(len(df_test)))


#Split features from label.
x_train=df_train.iloc[:,0:df.columns.size-1].to_numpy()
y_train=df_train.iloc[:,-1:].to_numpy()

x_test=df_test.iloc[:,0:df.columns.size-1].to_numpy()
y_test=df_test.iloc[:,-1:].to_numpy()

#Input shape
input_shape=(df.columns.size-1,1)
num_classes=2
print("Input shape:"+str(input_shape))
print("Num. classes: "+str(num_classes))

#Init accumulators.
mlp_acc=np.zeros(num_iter)
convnet_acc=np.zeros(num_iter)
mlp_bin_acc=np.zeros(num_iter)
convnet_bin_acc=np.zeros(num_iter)
mlp_fidelity=np.zeros(num_iter)
convnet_fidelity=np.zeros(num_iter)

for i in range(0,num_iter):
    print("*******************************************")
    print("ITERATION: " +str(i))

    #Define MLP.
    inputs=Input(shape=input_shape,name="input_layer")
    layer=Flatten()(inputs)
    layer=Dense(1024, activation='relu',name="hidden1")(layer)
    layer=Dense(1024, activation='relu',name="hidden2")(layer)
    layer=Dense(1024, activation='relu',name="hidden3")(layer)
    layer=Dense(num_classes-1, activation='sigmoid',name="output_layer")(layer)
    mlp = Model(inputs=inputs, outputs=layer)

    #Define convnet simple.
    inputs = Input(shape=input_shape)
    # In the first layer we only quantize the weights and not the input
    layer=Conv1D(128, 3,use_bias=False)(inputs)
    layer=BatchNormalization(momentum=0.999, scale=False)(layer)

    layer=Conv1D(128, 3, padding="same", use_bias=False )(layer)
    layer=MaxPool1D(pool_size=2, strides=2)(layer)
    layer=BatchNormalization(momentum=0.999, scale=False)(layer)

    layer=Conv1D(256, 3, padding="same", use_bias=False)(layer)
    layer=BatchNormalization(momentum=0.999, scale=False)(layer)

    layer=Conv1D(256, 3, padding="same", use_bias=False)(layer)
    layer=MaxPool1D(pool_size=2, strides=2)(layer)
    layer=BatchNormalization(momentum=0.999, scale=False)(layer)

    layer=Conv1D(512, 3, padding="same", use_bias=False)(layer)
    layer=BatchNormalization(momentum=0.999, scale=False)(layer)

    layer=Conv1D(512, 3, padding="same", use_bias=False)(layer)
    layer=MaxPool1D(pool_size=2, strides=2)(layer)
    layer=BatchNormalization(momentum=0.999, scale=False)(layer)
    layer=Flatten()(layer)

    layer=Dense(1024, use_bias=False)(layer)
    layer=BatchNormalization(momentum=0.999, scale=False)(layer)

    layer=Dense(1024, use_bias=False)(layer)
    layer=BatchNormalization(momentum=0.999, scale=False)(layer)

    layer=Dense(10, use_bias=False)(layer)
    layer=BatchNormalization(momentum=0.999, scale=False)(layer)
    layer=Dense(num_classes-1, activation="sigmoid")(layer)

    convnet=Model(inputs=inputs,outputs=layer)

    #Train originals
    mlp.compile(optimizer="adam",loss='binary_crossentropy', metrics=['accuracy'])
    mlp.fit(x_train, y_train, epochs=epoch_size, batch_size=batch_size,verbose=2)  
    print("MLP trained")
    convnet.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])
    convnet.fit(x_train, y_train, epochs=epoch_size, batch_size=batch_size,verbose=2)  
    print("CONVNET trained")

    #Create copies.
    mm=ModelManager(original_model=mlp,larq_configuration=larq_configuration)
    mlp_bin=mm.create_larq_model()    
    mlp_bin.fit(x_train, y_train, epochs=epoch_size, batch_size=batch_size,verbose=2)  
    print("MLP BIN trained")

    mm=ModelManager(original_model=convnet,larq_configuration=larq_configuration)
    convnet_bin=mm.create_larq_model()    
    convnet_bin.fit(x_train, y_train, epochs=epoch_size, batch_size=batch_size,verbose=2)  
    print("CONVNET BIN trained")

    mlp.save("mlp_shopping_i"+str(i)+str(larq_configuration["activation_relu"])+".h5")
    mlp_bin.save("mlp_bin_shopping_i"+str(i)+str(larq_configuration["activation_relu"])+".h5")
    convnet.save("convnet_shopping_i"+str(i)+str(larq_configuration["activation_relu"])+".h5")
    convnet_bin.save("convnet_bin_shopping_i"+str(i)+str(larq_configuration["activation_relu"])+".h5")

    #Test models.
    score = mlp.evaluate(x_test, y_test, verbose=2)
    mlp_acc[i]=score[1]
    print("MLP: Test loss:", score[0])
    print("MLP: Test accuracy:", score[1])

    score = convnet.evaluate(x_test, y_test, verbose=2)
    convnet_acc[i]=score[1]
    print("CONVNET: Test loss:", score[0])
    print("CONVNET: Test accuracy:", score[1])

    score = mlp_bin.evaluate(x_test, y_test, verbose=2)
    mlp_bin_acc[i]=score[1]
    print("MLP BIN: Test loss:", score[0])
    print("MLP BIN: Test accuracy:", score[1])

    score = convnet_bin.evaluate(x_test, y_test, verbose=2)
    convnet_bin_acc[i]=score[1]
    print("CONVNET BIN: Test loss:", score[0])
    print("CONVNET BIN: Test accuracy:", score[1])

    #Get fidelity.
    fidelity=Fidelity(original=mlp, surrogate=mlp_bin,x=x_test)    
    fidelity_value=fidelity.accuracy(last_layer="sigmoid")
    print("FIDELITY mlp vs mlp_bin" + str(fidelity_value))
    mlp_fidelity[i]=fidelity_value

    fidelity=Fidelity(original=convnet, surrogate=convnet_bin,x=x_test)
    fidelity_value=fidelity.accuracy(last_layer="sigmoid")
    print("FIDELITY convnet vs convnet_bin (SIMPLE)" + str(fidelity_value))
    convnet_fidelity[i]=fidelity_value


#Final results
print("FINAL RESULTS******************************")
print("MLP accuracy (mean,std): (" + str(np.mean(mlp_acc)) + "," + str(np.std(mlp_acc))+")")
print("CONVNET accuracy (mean,std):"+ str(np.mean(convnet_acc)) + "," + str(np.std(convnet_acc))+")") 
print("MLP_BIN accuracy (mean,std):"+ str(np.mean(mlp_bin_acc)) + "," + str(np.std(mlp_bin_acc))+")")
print("CONVNET_BIN accuracy (mean,std):"+ str(np.mean(convnet_bin_acc)) + "," + str(np.std(convnet_bin_acc))+")")
print("FIDELITY mlp vs mlp_bin (mean,std):" + str(np.mean(mlp_fidelity)) + "," + str(np.std(mlp_fidelity))+")") 
print("FIDELITY convnet vs convnet_bin (mean,std):" + str(np.mean(convnet_fidelity)) + "," + str(np.std(convnet_fidelity))+")")
