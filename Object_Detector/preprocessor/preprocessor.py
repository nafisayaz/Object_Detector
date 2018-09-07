

from tqdm import tqdm
import cv2           
import numpy as np
import os
from random import shuffle


TEST_DIR = '/home/nafis/projects/self_project/AI/dogsVScats/test_data/test'
IMG_SIZE = 50

class Preprocessor:
    
    def __init__(self):
        self.data = None


    def process_test_data(self):
        testing_data = []
        for img in tqdm(os.listdir(TEST_DIR)):
            path = os.path.join(TEST_DIR,img)
            img_num = img.split('.')[0]
            img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
            testing_data.append([np.array(img), img_num])
            
        shuffle(testing_data)
        np.save('test_data.npy', testing_data)
        return testing_data
