from model import load_FbDeepFace

class FaceRecognition:
    def __init__(self):
        self.model = load_FbDeepFace()
        self.database = './database'
        
    def predict(self, img):
        self.model.predict(img)
        
    def __compute_distance(self):
        
    
        