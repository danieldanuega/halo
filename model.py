from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout, LocallyConnected2D, Add, Activation
import os
from pathlib import Path
import zipfile
import gdown

def get_input_shape(model_name='DeepFace'):
    if model_name == 'DeepFace':
        shape = (152,152)
    elif model_name == 'DeepID':
        shape = (55,47)
        
    return shape

def download_model(file_path, output_path, url):
    home = str(Path.home())
    
    if os.path.isfile(file_path) != True:
        print("Model weights will be downloaded")
        if os.path.exists(home + output_path) == False:
            os.makedirs(home + output_path)
        
        link = url
        
        output = file_path + '.zip'
        
        gdown.download(link, output, quiet=False)
        
        with zipfile.ZipFile(output, 'r') as z:
            z.extractall(home + output_path)

def load_FbDeepFace():
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
    root = str(Path.home())
    model_path = root + '/.deepface/weights/VGGFace2_DeepFace_weights_val-0.9034.h5'
    download_model(file_path=model_path, output_path='/.deepface/weights', url='https://github.com/swghosh/DeepFace/releases/download/weights-vggface2-2d-aligned/VGGFace2_DeepFace_weights_val-0.9034.h5.zip')
            
    model.load_weights(model_path)
    
    fb_deepface_model = keras.Model(inputs=deepface_inputs, outputs=embedding_layer)
    
    return fb_deepface_model
    
def load_DeepId():
    deepid_inputs = keras.Input(shape=(55,47,3), name='In1')
    
    x = Conv2D(filters=20, kernel_size=(4,4), name='Conv1', activation='relu', input_shape=(55,47,3))(deepid_inputs)
    x = MaxPool2D(pool_size=2, strides=2, name='Pool1')(x)
    x = Dropout(rate=0.9, name='D1')(x)
    
    x = Conv2D(filters=40, kernel_size=(3,3), name='Conv2', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, name='Pool2')(x)
    x = Dropout(rate=0.9, name='D2')(x)
    
    x = Conv2D(filters=60, kernel_size=(3,3), name='Conv3', activation='relu')(x)
    x = MaxPool2D(pool_size=2, strides=2, name='Pool3')(x)
    x = Dropout(rate=0.9, name='D3')(x)
    
    x1 = Flatten()(x)
    fc11 = Dense(160, name='fc11')(x1)
    
    x2 = Conv2D(filters=80, kernel_size=(2,2), name='Conv4', activation='relu')(x)
    x2 = Flatten()(x2)
    fc12 = Dense(160, name='fc12')(x2)
    
    y = Add()([fc11, fc12])
    y = Activation('relu', name='deepid')(y)
    
    deepid_model = keras.Model(inputs=[deepid_inputs], outputs=y)
    
    root = str(Path.home())
    model_path = root + '/.deepface/weights/deepid_keras_weights.h5'
    download_model(file_path=model_path, output_path='/.deepface/weights', url='https://1drv.ms/u/s!Ai3f17Uo85P5gtwEZxv9Nbprv3G4Jw')
    
    deepid_model.load_weights(model_path)
    
    return deepid_model
