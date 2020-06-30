from model import load_FbDeepFace
import os
import pickle
from tqdm import tqdm
import helper
import pandas as pd
import numpy as np

class FaceRecognition:
    def __init__(self, database='./database'):
        self.model = load_FbDeepFace()
        self.database = database
        
        # Check if the representations of employee faces is exist or not
        if os.path.isdir(self.database) == True:
            file_name = 'representations.pkl'
            file_name = file_name.replace("-", "_").lower()
            
            isTrainAgain = False
            
            if os.path.exists(os.path.join(self.database, file_name)):
                f = open(os.path.join(self.database, file_name), 'rb')
                try:
                    representations = pickle.load(f)
                except EOFError:
                    print("representations.pkl seems empty")
                    
                _, counts = self.__count_files(self.database)
                
                # If representations.pkl exist but there are new employees or resign employees
                if len(representations) != counts:
                    print('In database: {}'.format(counts))
                    print('In representations.pkl: {}'.format(len(representations)))
                    print('Found new employees or one of them have resign')
                    print('Begin analyzing')
                    isTrainAgain = True
                else:
                    self.representations = representations
                    print('There are {} of {} faces found in the database'.format(len(self.representations), counts))
            
            # Find the employees face representation as vector
            if isTrainAgain or os.path.exists(os.path.join(self.database, file_name)) == False:
                employees, _ = self.__count_files(self.database)
                
                if len(employees) == 0:
                    raise ValueError("There is no image in ", self.database," folder!")
                
                #------------------------
                #find representations for db images
                
                representations = []
                
                pbar = tqdm(range(0,len(employees)))
                
                #for employee in employees:
                for index in pbar:
                    employee = employees[index]
                    
                    pbar.set_description('Finding embedding for {}'.format(employee))
                    
                    shape = self.model.layers[0].input_shape
                    
                    input_shape = shape[0][1:3] if type(shape) is list else shape[1:3]
                    input_shape_x = input_shape[0]; input_shape_y = input_shape[1]
                    
                    img = helper.detectFace(employee, (input_shape_y, input_shape_x), enforce_detection = True)
                    representation = self.model.predict(img)[0,:]
                    
                    instance = []
                    instance.append(employee)
                    instance.append(representation)
                        
                    #-------------------------------
                    
                    representations.append(instance)
                
                f = open(self.database+'/'+file_name, "wb")
                pickle.dump(representations, f)
                f.close()
                
                self.representations = representations
                
                print("Representations stored in ",self.database,"/",file_name)
        else:
            raise ValueError("database not a directory")
        
    def predict(self, img):
        if self.representations == None or len(self.representations) == 0:
            raise AttributeError("Representations file not loaded correctly")
        
        df = pd.DataFrame(self.representations, columns=['identity', 'representation'])
        
        target_representation = self.model.predict(img)[0,:]
        
        distances = []
        for index, col in df.iterrows():
            source_representation = col['representation']
            # distance = self.__euclideanDistance(self.__l2_normalize(source_representation), self.__l2_normalize(target_representation))
            # distance = self.__cosineDistance(source_representation, target_representation)
            distance = self.__euclideanDistance(source_representation, target_representation)
            distances.append(distance)
        
        # threshold = helper.findThreshold('DeepFace', 'euclidean_l2')
        # threshold = helper.findThreshold('DeepFace', 'cosine')
        threshold = helper.findThreshold('DeepFace', 'euclidean')
        
        df['distances'] = distances
        df = df.drop(columns=['representation'])
        df = df[df.distances <= threshold]

        df = df.sort_values(by=['distances'], ascending=True).reset_index(drop=True)
        print(df)
        
        if df.empty:
            return "", ""
        
        person = df.iloc[0]['identity']
        confidence = df.iloc[0]['distances']
        folder, sep, name_imageName = person[2::].partition('/')
        name, sep, imageName = name_imageName.partition('/')
        return name.capitalize(), confidence
    
    def __cosineDistance(self, origin, test):
        a = np.matmul(np.transpose(origin), test)
        b = np.sum(np.multiply(origin, origin))
        c = np.sum(np.multiply(test, test))
        return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
    
    def __count_files(self, dir_path):
        count = 0
        items = []
        for root, directory, files in os.walk(dir_path):
            for f in files:
                if '.jpg' in f:
                    count+=1
                    exact_path = root + '/' + f
                    items.append(exact_path)
        
        return (items, count)
    
    def __l2_normalize(self, x):
        return x / np.sqrt(np.sum(np.multiply(x, x)))
 
    def __euclideanDistance(self, source_representation, test_representation):
        euclidean_distance = source_representation - test_representation
        euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
        euclidean_distance = np.sqrt(euclidean_distance)
        return euclidean_distance
    