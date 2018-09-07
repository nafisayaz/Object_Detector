

import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm

TRAIN_DIR = '/home/nafis/projects/self_project/AI/dogsVScats/data/train_data/train'
IMG_SIZE = 50

class Train:

	def __init__(self):
		self._data = None

	def label_img(self, img):
	    word_label = img.split('.')[-3]
	    # conversion to one-hot array [cat,dog]
	    #                            [much cat, no dog]
	    if word_label == 'cat': return [1,0]
	    #                             [no cat, very doggo]
	    elif word_label == 'dog': return [0,1]


	def create_train_data(self):
	    training_data = []
	    for img in tqdm(os.listdir(TRAIN_DIR)):
	        label = self.label_img(img)
	        path = os.path.join(TRAIN_DIR,img)
	        img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
	        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))
	        training_data.append([np.array(img),np.array(label)])
	    shuffle(training_data)
	    np.save('train_data.npy', training_data)
	    return training_data

