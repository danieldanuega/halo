import os
from os.path import join, exists
from os import mkdir, listdir
import numpy as np
import matplotlib.pyplot as plt
from model import get_model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping


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
        self.BATCH_SIZE = 32
        self.NUMBER_OF_TRAINING_IMAGES = 250
        self.NUMBER_OF_TESTING_IMAGES = 50
        self.IMAGE_HEIGHT = 224
        self.IMAGE_WIDTH = 224
        self.model = get_model()
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
            # horizontal_flip=True,
            fill_mode="nearest",
            # zoom_range=0.3,
            # width_shift_range=0.3,
            # height_shift_range=0.3,
            rotation_range=30
        )
        return img_data_generator

    def training(self):
        self.training_generator = FaceRecognition.data_generator().flow_from_directory(
            self.TRAINING_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            batch_size=self.BATCH_SIZE,
            class_mode='categorical'
        )

        testing_generator = FaceRecognition.data_generator().flow_from_directory(
            self.TESTING_DATA_DIRECTORY,
            target_size=(self.IMAGE_WIDTH, self.IMAGE_HEIGHT),
            class_mode='categorical'
        )
        
        early_stop = EarlyStopping(monitor='val_loss',patience=3)

        self.model.compile(
            loss='categorical_crossentropy',
#             optimizer=optimizers.SGD(lr=1e-4, momentum=0.9, decay=1e-2 / self.EPOCHS),
            optimizer='adam',
            metrics=["accuracy"]
        )

        history = self.model.fit(
            self.training_generator,
            # steps_per_epoch=self.NUMBER_OF_TRAINING_IMAGES//self.BATCH_SIZE,
            epochs=self.EPOCHS,
            validation_data=testing_generator,
            shuffle=True,
            # validation_steps=2
            callbacks=[early_stop]
        )

        FaceRecognition.plot_training(history)

    def save_model(self, model_name):
        model_path = "./model"
        if not os.path.exists(model_path):
            os.mkdir(model_path)

        self.model.save(os.path.join(model_path, model_name))
        class_names = self.training_generator.class_indices
        class_names_file_reverse = model_name[:-3] + "_class_names_reverse.npy"
        class_names_file = model_name[:-3] + "_class_names.npy"
        np.save(os.path.join(model_path, class_names_file_reverse), class_names)
        class_names_reversed = np.load(os.path.join(model_path, class_names_file_reverse), allow_pickle=True).item()
        class_names = dict([(value, key) for key, value in class_names_reversed.items()])
        np.save(os.path.join(model_path, class_names_file), class_names)

    @staticmethod
    def load_saved_model(model_path):
        model = load_model(model_path)
        return model

    @staticmethod
    def model_prediction(image, face_array, face, model_path, class_names_path):
        class_name = "None Class Name"
        # face_array, face = get_detected_face(image)
        model = load_model(model_path)
        face_array = face_array.astype('float32')
        input_sample = np.expand_dims(face_array, axis=0)
        result = model.predict(input_sample)
        result = np.argmax(result, axis=1)
        index = result[0]

        classes = np.load(class_names_path, allow_pickle=True).item()
        # print(classes, type(classes), classes.items())
        if type(classes) is dict:
            for k, v in classes.items():
                if k == index:
                    class_name = v

        return class_name
    
# Call to train the model
if __name__ == '__main__':
    model_name = "face_recognition.h5"
    faceRecognition = FaceRecognition()
    faceRecognition.training()
    faceRecognition.save_model(model_name)
