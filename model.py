import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers

#################### Load the trained CNN model  ###################################
def loss_f(y_true,y_pred):
    loss = tf.keras.losses.logcosh(y_true,y_pred)
    return loss

def cnn_model(input_shape):
    img_input = layers.Input(input_shape)
    ## First CNN block:
    conv2D_1 = layers.Conv2D(64,(3,3),strides = 2, padding = 'valid', name = 'Conv_1',input_shape = input_shape, kernel_initializer = 'he_normal')(img_input)
    conv2D_1 = layers.BatchNormalization(axis = 3)(conv2D_1)
    conv2D_1 = layers.PReLU()(conv2D_1)
    conv2D_2 = layers.Conv2D(64,(3,3),strides = 1, padding = 'same',name = 'Conv_2',kernel_initializer = 'he_normal')(conv2D_1)
    conv2D_2 = layers.BatchNormalization(axis = 3)(conv2D_2)
    conv2D_2 = layers.PReLU()(conv2D_2)
    maxPooling_2 = layers.MaxPooling2D((3,3),strides = (2,2),name = 'MaxPooling_1')(conv2D_2)

    ## Second CNN block:
    conv2D_3 = layers.Conv2D(128,(3,3),strides = 1, padding = 'same', name = 'Conv_3',kernel_initializer = 'he_normal')(maxPooling_2)
    conv2D_4 = layers.BatchNormalization(axis = 3)(conv2D_3)
    conv2D_3 = layers.PReLU()(conv2D_3)
    conv2D_4 = layers.Conv2D(128,(3,3),strides = 1, padding = 'same',name = 'Conv_4',kernel_initializer = 'he_normal')(conv2D_3)
    conv2D_4 = layers.BatchNormalization(axis = 3)(conv2D_4)
    conv2D_4 = layers.PReLU()(conv2D_4)
    averagePooling_4 = layers.AveragePooling2D((3,3),strides = (2,2),name = 'AveragePooling_1')(conv2D_4)
    
    flatten = layers.Flatten()(averagePooling_4)
    fc = layers.Dropout(0.5)(flatten)
    fc = layers.Dense(256,kernel_regularizer=regularizers.l2(0.01),kernel_initializer = 'he_normal',name = 'FC1')(fc)
    fc = layers.Activation('relu')(fc)
    fc = layers.Dropout(0.3)(fc)
    fc = layers.Dense(128,kernel_regularizer=regularizers.l2(0.01),kernel_initializer = 'he_normal',name = 'FC2')(fc)
    fc = layers.Activation('relu')(fc)
    fc = layers.Dropout(0.3)(fc)
    output = layers.Dense(1)(fc)
    
    cnn_model = models.Model(inputs = [img_input], outputs = [output] )
    
    return cnn_model


def model_load():

    file_name = "./0413_3.hdf5"
    trained_model = cnn_model([60,100,2])
    trained_model.load_weights(file_name)
    trained_model.compile(optimizer = 'adam',loss = loss_f, metrics= ['mae','mape'])
    return trained_model


################################################################################

class Model(object):
     

    def __init__(self):
        self.model = model_load()

    def model_pred(self,image_pair):
        return self.model.predict(image_pair)
