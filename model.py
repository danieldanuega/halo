from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten

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

    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu')) 
    model.add(Dense(2, activation='softmax'))
    
    return model

