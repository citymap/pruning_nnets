import tensorflow as tf  
import numpy as np
from keras.models import save_model, load_model

# function to save weights as .npy files for loading into feedforward network
def save_weights_and_biases(prunable_array, name):
	for i in range(4):
		layer_num = i + 1
		path_name = 'weights/' + name + '-' + str(layer_num)
		np.save(path_name, prunable_array[i])

# loading data from mnist	
mnist = tf.keras.datasets.mnist  
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# normalizing input data (pixels now range from 0 to 1)
x_train = x_train / 255.0  
x_test = x_test / 255.0  

# neural network structure
layer1 = tf.keras.layers.Flatten()
layer2 = tf.keras.layers.Dense(1000, activation=tf.nn.relu) # fully-connected layer with 
layer3 = tf.keras.layers.Dense(1000, activation=tf.nn.relu) # relu activation function
layer4 = tf.keras.layers.Dense(500, activation=tf.nn.relu)
layer5 = tf.keras.layers.Dense(200, activation=tf.nn.relu)
logits_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

# adding layers to model
model = tf.keras.models.Sequential() 
model.add(layer1)  
model.add(layer2)  
model.add(layer3)  
model.add(layer4) 
model.add(layer5)
model.add(logits_layer) 

# model compiled to use 'adam' optimizer, and loss 'sparse_categorical_crossentropy'
model.compile(optimizer='adam',  
              loss='sparse_categorical_crossentropy',  
              metrics=['accuracy']) 

# model trained for 3 epochs
# epoch = 1 complete round of the dataset (in this case, x_train and y_train)
model.fit(x_train, y_train, epochs=3) 

# get weights/biases into an array. 
# NOTE: biases will not be used in forward pass (pruning_tf.py), but are returned as default in Keras
prunable_weights = [layer2.get_weights()[0], layer3.get_weights()[0], layer4.get_weights()[0], layer5.get_weights()[0], logits_layer.get_weights()[0]]
prunable_biases = [layer2.get_weights()[1], layer3.get_weights()[1], layer4.get_weights()[1], layer5.get_weights()[1], logits_layer.get_weights()[1]]

# save weights/biases in .npy format to weights/ folder
save_weights_and_biases(prunable_weights, 'weight')
save_weights_and_biases(prunable_biases, 'bias')

# validation loss and accuracy to test quality of model
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss)
print(val_acc)
