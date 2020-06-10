from tensorflow.keras import models
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras import regularizers

def get_model():
    model = models.Sequential()
    
    model.add(Conv2D(filters=128, kernel_size=(4,4),
                     activation='relu',
                     kernel_regularizer=regularizers.l2(0.),
                     activity_regularizer=regularizers.l2(0.),
                     input_shape=(224, 224, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.05))
    model.add(Conv2D(filters=256, kernel_size=(4,4), activation='relu',
                     kernel_regularizer=regularizers.l2(0.),
                     activity_regularizer=regularizers.l2(0.)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.05))
    model.add(Conv2D(512, kernel_size=(4,4), activation='relu',
                     kernel_regularizer=regularizers.l2(0.),
                     activity_regularizer=regularizers.l2(0.)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(BatchNormalization())
    model.add(Dropout(0.05))
    
    model.add(Flatten())
    
    model.add(Dense(256, activation='relu', # 3000
                    kernel_regularizer=regularizers.l2(0.),
                    activity_regularizer=regularizers.l2(0.)))
    model.add(Dropout(0.05))
    model.add(Dense(2, activation='softmax'))
    return model
    
