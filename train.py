import os
import re
from os.path import join, exists
from os import mkdir, listdir
import numpy as np
import matplotlib.pyplot as plt
from model import get_model
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


import glob
# from mtcnn import MTCNN
from PIL import Image
# import cv2

# def get_detected_face(img, required_size=(224, 224)):
#     detector = MTCNN()
#     results = detector.detect_faces(img)
#     for face in results:
#         x, y, width, height = face['box']
#         face = img[y:y + height, x:x + width]
#         image = Image.fromarray(face)
#         image = image.resize(required_size)
#         face_array = np.asarray(image)
#     return face_array, face


class FaceRecognition:

    def __init__(self):
        self.TRAINING_DATA_DIRECTORY = "./dataset/train"
        self.TESTING_DATA_DIRECTORY = "./dataset/test"
        self.EPOCHS = 50
        self.BATCH_SIZE = 16
        self.NUMBER_OF_TRAINING_IMAGES = 250
        self.NUMBER_OF_TESTING_IMAGES = 50
        self.IMAGE_HEIGHT = 224
        self.IMAGE_WIDTH = 224
        self.model = get_model()
        self.MODEL_PATH = "./models/model5"
        self.training_generator = None

    @staticmethod
    def plot_training(history):
        plot_folder = "plot"
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.1, 1])
        plt.legend(loc='lower right')

        if not os.path.exists(plot_folder):
            os.mkdir(plot_folder)

        plt.savefig(os.path.join(plot_folder, "model_accuracy.png"))

    @staticmethod
    def data_generator():
        img_data_generator = ImageDataGenerator(
            rescale=1./255,
            fill_mode="nearest",
            width_shift_range=0.3,
            height_shift_range=0.3,
            rotation_range=30
        )
        return img_data_generator

    def training(self):
        self.training_generator = FaceRecognition.data_generator().flow_from_directory(
            self.TRAINING_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            batch_size=self.BATCH_SIZE,
            color_mode='rgb',
            class_mode='categorical'
        )
        
        print(self.training_generator.class_indices)

        testing_generator = FaceRecognition.data_generator().flow_from_directory(
            self.TESTING_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical',
            color_mode='rgb',
            shuffle=False
        )
        
        early_stop = EarlyStopping(monitor='val_loss',patience=3)
        checkpoint = ModelCheckpoint(os.path.join(self.MODEL_PATH, 'best_face_recognition.h5'), monitor='val_loss', verbose=1, save_best_only=True)

        self.model.compile(
            loss='categorical_crossentropy',
#             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-2 / self.EPOCHS),
            optimizer='adam',
            metrics=["accuracy"]
        )

        history = self.model.fit(
            self.training_generator,
            epochs=self.EPOCHS,
            validation_data=testing_generator,
            callbacks=[early_stop, checkpoint]
        )

#         FaceRecognition.plot_training(history)

    def save_model(self, model_name, lite_model_name):
        model_path = self.MODEL_PATH
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        
        # Save the latest model in last epochs
        self.model.save(os.path.join(model_path, model_name))
        class_names = self.training_generator.class_indices
        class_names_file_reverse = model_name[:-3] + "_class_names_reverse.npy"
        class_names_file = model_name[:-3] + "_class_names.npy"
        np.save(os.path.join(model_path, class_names_file_reverse), class_names)
        class_names_reversed = np.load(os.path.join(model_path, class_names_file_reverse), allow_pickle=True).item()
        class_names = dict([(value, key) for key, value in class_names_reversed.items()])
        np.save(os.path.join(model_path, class_names_file), class_names)
        # Save to tensorflow lite format .tflite for both latest and best model
        converter1 = tf.lite.TFLiteConverter.from_keras_model_file(os.path.join(model_path, model_name))
        converter2 = tf.lite.TFLiteConverter.from_keras_model_file(os.path.join(model_path, 'best_face_recognition.h5'))
        tflite_model = converter1.convert()
        best_tflite_model = converter2.convert()
        open(os.path.join(model_path, lite_model_name), 'wb').write(tflite_model)
        open(os.path.join(model_path, 'best_lite_face_recognition.tflite'), 'wb').write(best_tflite_model)

    @staticmethod
    def load_saved_model(model_path):
        model = load_model(model_path)
        return model

    @staticmethod
    def model_prediction(face_array, model_path, class_names_path, threshold):
        class_name = "I don't know yet"
        face_array = face_array.astype('float32')
        input_sample = np.expand_dims(face_array, axis=0)
        # Check if use tensorflow or tensorflow lite
        if re.search('.h5', model_path):
            model = load_model(model_path)
            results = model.predict(input_sample)
            results = np.argmax(results, axis=1)
            index = results[0]
        elif re.search('.tflite', model_path):
            interpreter = tf.lite.Interpreter(model_path=model_path)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            interpreter.set_tensor(input_details[0]['index'], input_sample)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            # Squeeze 2D array into 1D array
            results = np.squeeze(output_data)
            # Sort the probabilities array from max to min, but here we use the index as the representation
            indices = results.argsort()[-5:][::-1]
            # Get the index of max probabilities
            index = indices[0]
            print('probabilites: {}'.format(results))
            print(results[index])
            if results[index] < threshold:
                return class_name
        else:
            return class_name

        classes = np.load(class_names_path, allow_pickle=True).item()
        print(classes)
        if type(classes) is dict:
            for k, v in classes.items():
                if k == index:
                    class_name = v

        return class_name
    
# Call to train the model
if __name__ == '__main__':
    model_name = "face_recognition.h5"
    lite_model_name = "lite_face_recognition.tflite"
    faceRecognition = FaceRecognition()
    faceRecognition.training()
    faceRecognition.save_model(model_name, lite_model_name)
