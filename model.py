from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, LocallyConnected2D, Add, Activation
import os
from pathlib import Path
import zipfile
import gdown

def get_input_shape(model_name='deepface'):
    if model_name == 'deepface':
        shape = (152,152)
    elif model_name == 'deepid':
        shape = (47,55)
        
    return shape

def load_FbDeepFace():
    # model = keras.Sequential()
    
    # model.add(Conv2D(filters=32, kernel_size=(11,11), activation='relu', name='C1', input_shape=(152,152,3)))
    # model.add(MaxPool2D(pool_size=(3,3), strides=2, padding='same', name='M2'))
    # model.add(Conv2D(filters=16, kernel_size=(9,9), activation='relu', name='C3'))
    
    # model.add(LocallyConnected2D(filters=16, kernel_size=(9,9), activation='relu', name='L4'))
    # model.add(LocallyConnected2D(filters=16, kernel_size=(7,7), strides=2, activation='relu', name='L5'))
    # model.add(LocallyConnected2D(filters=16, kernel_size=(5,5), activation='relu', name='L6'))
    
    # model.add(Flatten(name='F0'))
    
    # model.add(Dense(4096, activation='relu', name='F7'))
    # model.add(Dropout(rate=0.5, name='D0'))
    # model.add(Dense(8631, activation='softmax', name='F8'))
    
    # =====================================================================================
    
    deepface_inputs = keras.Input(shape=(152,152,3), name="In1")
    x = Conv2D(filters=32, kernel_size=(11,11), activation='relu', name='C1')(deepface_inputs)
    x = MaxPool2D(pool_size=(3,3), strides=2, padding='same', name='M2')(x)
    x = Conv2D(filters=16, kernel_size=(9,9), activation='relu', name='C3')(x)
    
    x = LocallyConnected2D(filters=16, kernel_size=(9,9), activation='relu', name='L4')(x)
    x = LocallyConnected2D(filters=16, kernel_size=(7,7), strides=2, activation='relu', name='L5')(x)
    x = LocallyConnected2D(filters=16, kernel_size=(5,5), activation='relu', name='L6')(x)
    
    flat = Flatten(name='F0')(x)
    
    embedding_layer = Dense(4096, activation='relu', name='F7')(flat)
    drop_1 = Dropout(rate=0.5, name='D0')(embedding_layer)
    deepface_outputs = Dense(8631, activation='softmax', name='F8')(drop_1)
    model = keras.Model(inputs=deepface_inputs, outputs=deepface_outputs)
    
    # Download the weights
    home = str(Path.home())
    model_path = home + '/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5'
    
    if os.path.isfile(model_path) != True:
        print("DeepFace weights by will be downloaded ")
        os.makedirs(home + '/.deepface/weights')
        
        url = 'https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip'
        
        output = model_path + '.zip'
        
        gdown.download(url, output, quiet=False)
        
        with zipfile.ZipFile(output, 'r') as z:
            z.extractall(home + '/.deepface/weights')
            
    model.load_weights(model_path)
    
    fb_deepface_model = keras.Model(inputs=deepface_inputs, outputs=embedding_layer)
    
    return fb_deepface_model
    
    def load_DeepId():
        deepid_inputs = keras.Input(shape=(55,47,3), name='In1')
        
        x = Conv2D(filters=20, kernel_size=(4,4), name='Conv1', activation='relu', input_shape=(55,47,3))(deepid_inputs)
        x = MaxPool2D(pool_size=2, strides=2, name='Pool1')(x)
        x = Dropout(rate=1, name='D1')(x)
        
        x = Conv2D(filters=40, kernel_size=(3,3), name='Conv2', activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2, name='Pool2')(x)
        x = Dropout(rate=1, name='D2')(x)
        
        x = Conv2D(filters=60, kernel_size=(3,3), name='Conv3', activation='relu')(x)
        x = MaxPool2D(pool_size=2, strides=2, name='Pool3')(x)
        x = Dropout(rate=1, name='D3')(x)
        
        x1 = Flatten()(x)
        fc11 = Dense(160, name='fc11')(x1)
        
        x2 = Conv2D(filters=80, kernel_size=(2,2), name='Conv4', activation='relu')(x)
        x2 = Flatten()(x2)
        fc12 = Dense(160, name='fc12')(x2)
        
        y = Add()([fc11, fc12])
        y = Activation('relu', name='deepid')(y)
        
        model = keras.Model(inputs=[deepface_inputs], outputs=y)
        
        # TODO: Downloads the weights and load it!