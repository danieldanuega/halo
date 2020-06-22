from model import load_FbDeepFace
import os
import pickle
from tqdm import tqdm
import functions

class FaceRecognition:
    def __init__(self, database='./database'):
        self.model = load_FbDeepFace()
        self.database = database
        
        # Check if the representations of employees face is exist or not
        if os.path.isdir(self.database) == True:
            file_name = 'representations.pkl'
            file_name = file_name.replace("-", "_").lower()
            
            isTrainAgain = False
            
            if os.path.exists(os.path.join(self.database, file_name)):
                f = open(os.path.join(self.database, file_name), 'rb')
                representations = pickle.load(f)
                
                # If representations exist but there are new employees or resign employees
                if len(representations) != len(os.listdir(self.database))-1:
                    print('Found new employees or one of them have resign')
                    print('Begin analyzing')
                    isTrainAgain = True
                else:
                    print('There are {} faces found in the database'.format(len(representations)))
            
            # Find the employees face representation as vector
            if isTrainAgain or os.path.exists(os.path.join(self.database, file_name)) == False:
                employees = []
            
                for root, directory, files in os.walk(self.database):
                    for f in files:
                        if ('.jpg' in f):
                            exact_path = root + "/" + f
                            employees.append(exact_path)
                
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
                    
                    img = functions.detectFace(employee, (input_shape_y, input_shape_x), enforce_detection = True)
                    representation = self.model.predict(img)[0,:]
                    
                    instance = []
                    instance.append(employee)
                    instance.append(representation)
                        
                    #-------------------------------
                    
                    representations.append(instance)
                
                f = open(self.database+'/'+file_name, "wb")
                pickle.dump(representations, f)
                f.close()
                
                print("Representations stored in ",self.database,"/",file_name)
        else:
            raise ValueError("database not a directory")
        
    def predict(self, img):
        return 0
        
    def __compute_distance(self):
        return 0
    