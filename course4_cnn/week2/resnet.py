import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from resnets_utils import *
from keras.initializers import glorot_uniform
import scipy.misc
from matplotlib.pyplot import imshow

import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)

def identity_block(X, f, filters, stage, block):
    # stage = int used to name layers
    # block = string used to name layers
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    # First layer
    # Perform 1x1 conv on X
    X = Conv2D(filters=F1, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base + '2a', kernel_initializer=glorot_uniform(seed=0))(X)   
    # Perform batch normalization
    X = BatchNormalization(axis=3, name=bn_name_base+'2a')(X)
    X = Activation('relu')(X)

    # Second layer
    X = Conv2D(filters=F2, kernel_size=(f,f), strides=(1,1), padding='same', name=conv_name_base + '2b', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third layer
    X = Conv2D(filters=F3, kernel_size=(1,1), strides=(1,1), padding='valid', name=conv_name_base+'2c', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)
    # Don't compute activation yet! We need to add the original activation in skip connection

    # Skip connection
    X = Activation('relu')(Add()([X, X_shortcut]))

def convolutional_block(X, f, filters, stage, block, s=2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    # First layer
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second layer
    X = Conv2D(F2, (f,f), strides=(1,1), name=conv_name_base+'2b', padding='same', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    # Third layer
    X = Conv2D(F3, (1,1), strides=(1,1), name=conv_name_base+'2c', padding='valid', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base+'2c')(X)
    
    # Skip connection
    # Use strides (s,s) to shrink height & width and F3 filters to match channel
    X_shortcut = Conv2D(F3, (1,1), strides=(s,s), name=conv_name_base+'1', padding='valid', kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base+'1')(X_shortcut)

    return Activation('relu')(Add()([X, X_shortcut]))
    
def ResNet50(input_shape=(64,64,3), classes=6):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3,3))(X_input)

    X = Conv2D(64, (7,7), strides=(2,2), name='conv1', kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name='bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3,3), strides=(2,2))(X)

    X = convolutional_block(X, f=3, filters=[64,64,256], stage=2, block='a', s=1)
    X = identity_block(X, 3, [64,64,256], stage=2, block='b')
    X = identity_block(X, 3, [64,64,256], stage=2, block='c')

    X = convolutional_block(X, f=3, filters=[128,128,512], stage=3, block='a', s=2)
    for i in range(3):
        X = identity_block(X, 3, [128,128,512], stage=3, block=chr(ord('b') + i))

    X = convolutional_block(X, f=3, filters=[256,256,1024], stage=4, block='a', s=2)
    for i in range(5):
        X = identity_block(X, 3, [256,256,1024], stage=4, block=chr(ord('b') + i))
    
    X = convolutional_block(X, f=3, filters=[512,512,2048], stage=5, block='a', s=2)
    for i in range(2):
        X = identity_block(X, 3, [512,512,2048], stage=5, block=chr(ord('b') + i))

    X = AveragePooling2D((2,2), name='avg_pool')(X)

    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer=glorot_uniform(seed=0)(X))

    model = Model(inputs=X_input, outputs=X, name='ResNet50')

    return model
    
model = ResNet50(input_shape=(64,64,3), classes=6)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T

model.fit(X_train, Y_train, epochs=2, batch_size=32)

preds = model.evaluate(X_test, Y_test)