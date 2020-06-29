import tensorflow as tf
import os
from model import load_FbDeepFace

MODEL_VER = '1'
SAVE_PATH = f'{os.getcwd()}/models/tf_FbDeepFace'

if os.path.exists(SAVE_PATH) == False:
    os.makedirs(SAVE_PATH)
    print('Path created!')

print('Load model . . .')
model = load_FbDeepFace()

tf.saved_model.save(model, export_dir=os.path.join(SAVE_PATH,MODEL_VER))
print('Model ver {} has been successfully saved!'.format(MODEL_VER))
