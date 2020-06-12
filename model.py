from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten, Lambda
from tensorflow.keras.regularizers import l2
import tensorflow as tf

def get_model():
    model = Sequential()

    model.add(Conv2D(filters=128, kernel_size=(11,11), activation='relu', input_shape=(224,224,3)))
    model.add(MaxPool2D(pool_size=(3,3)))
    model.add(Conv2D(filters=256, kernel_size=(5,5), activation='relu'))
    model.add(MaxPool2D(pool_size=(3,3)))

    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))
    model.add(Conv2D(filters=256, kernel_size=(3,3), activation='relu'))

    model.add(MaxPool2D(pool_size=(3,3)))

    model.add(Flatten())

    model.add(Dense(256, activation=None))
    model.add(Lambda(lambda x: tf.math.l2_normalize(x, axis=1)))
    
    # SVM
    model.add(Dense(2, activation='linear', kernel_regularizer=l2(0.0001)))
    
    return model

