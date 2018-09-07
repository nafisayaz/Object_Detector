
import tensorflow as tf
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import os

IMG_SIZE = 50
LR = 1e-3


MODEL_NAME = 'dogsvscats-{}-{}.model'.format(LR, '2conv-basic')

class MODEL_DNN:
	
	def __init__(self):
		self.data = None
		
	
	def create_model(self):

		tf.reset_default_graph()
		convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

		convnet = conv_2d(convnet, 32, 5, activation='relu')
		convnet = max_pool_2d(convnet, 5)

		convnet = conv_2d(convnet, 64, 5, activation='relu')
		convnet = max_pool_2d(convnet, 5)

		convnet = conv_2d(convnet, 128, 5, activation='relu')
		convnet = max_pool_2d(convnet, 5)

		convnet = conv_2d(convnet, 64, 5, activation='relu')
		convnet = max_pool_2d(convnet, 5)

		convnet = conv_2d(convnet, 32, 5, activation='relu')
		convnet = max_pool_2d(convnet, 5)

		convnet = fully_connected(convnet, 1024, activation='relu')
		convnet = dropout(convnet, 0.8)

		convnet = fully_connected(convnet, 2, activation='softmax')
		convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

		model = tflearn.DNN(convnet, tensorboard_dir='log')

	def load_model(self, model):
		if os.path.exists('/home/nafis/projects/self_project/AI/dogsVScats/{}.meta'.format(MODEL_NAME)):
		    model.load(MODEL_NAME)
		    print('model loaded!')

	def fit_model(self, train_data, model):

		train = train_data[:-500]
		test = train_data[-500:]

		X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
		Y = [i[1] for i in train]

		test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
		test_y = [i[1] for i in test]

		model.fit({'input': X}, {'targets': Y}, n_epoch=10, validation_set=({'input': test_x}, {'targets': test_y}), 
		    snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

		model.save(MODEL_NAME)

