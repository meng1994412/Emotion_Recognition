# import packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras.regularizers import l2
from keras import backend as K

class EmotionVGGNet:
    @staticmethod
    def build(width, height, depth, classes, reg = 0.0005):
        # initialize the model along with the input shape to be
        # "channels last" and the channels dimension itself
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        # if we use "channels first", update the input shape and channels dimensions
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        # Block #1: first CONV => ELU => CONV => ELU => POOL layer set
        model.add(Conv2D(32, (3, 3), padding = "same",
            kernel_initializer = "he_normal", kernel_regularizer = l2(reg),
            input_shape = inputShape))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(32, (3, 3), padding = "same", kernel_regularizer = l2(reg),
            kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # Block #2: second CONV => ELU => CONV => ELU => POOL layer set
        model.add(Conv2D(64, (3, 3), padding = "same", kernel_regularizer = l2(reg),
            kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(64, (3, 3), padding = "same", kernel_regularizer = l2(reg),
            kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # Block #3: third CONV => ELU => CONV => ELU => POOL
        model.add(Conv2D(128, (3, 3), padding = "same", kernel_regularizer = l2(reg),
            kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(128, (3, 3), padding = "same", kernel_regularizer = l2(reg),
            kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # Block #3*: fourth CONV => ELU => CONV => ELU => POOL
        # uncomment this block for experiment #5 or later
        model.add(Conv2D(256, (3, 3), padding = "same", kernel_regularizer = l2(reg),
            kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Conv2D(256, (3, 3), padding = "same", kernel_regularizer = l2(reg),
            kernel_initializer = "he_normal"))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(MaxPooling2D(pool_size = (2, 2)))
        model.add(Dropout(0.25))

        # Block #4: first set of FC => ELU layer set
        model.add(Flatten())
        model.add(Dense(64, kernel_initializer = "he_normal",
            kernel_regularizer = l2(reg)))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))

        # Block #5: second set of FC => ELU layer set
        model.add(Dense(64, kernel_initializer = "he_normal",
            kernel_regularizer = l2(reg)))
        model.add(ELU())
        model.add(BatchNormalization(axis = chanDim))
        model.add(Dropout(0.5))

        # Block #6: softmax classifier
        model.add(Dense(classes, kernel_initializer = "he_normal",
            kernel_regularizer = l2(reg)))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model
