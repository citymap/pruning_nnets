import numpy as np
import tensorflow as tf 
from pruning_functions import pruning_func

# main function to detect accuracy across test set with pruned weights
def perform_forward_pass(pruning_bool, pruned_weights):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_test = x_test.reshape(x_test.shape[0], 28 * 28) / 255.0

	# if pruning, we load pruned weights. Otherwise, we load unpruned weights
	if pruning_bool:
		w1 = tf.Variable(initial_value = pruned_weights[0]) # pruned weights
		w2 = tf.Variable(initial_value = pruned_weights[1])
		w3 = tf.Variable(initial_value = pruned_weights[2])
		w4 = tf.Variable(initial_value = pruned_weights[3])
	else:
		w1 = tf.Variable(initial_value = np.load('weights/weight-1.npy')) # unpruned weights
		w2 = tf.Variable(initial_value = np.load('weights/weight-2.npy'))
		w3 = tf.Variable(initial_value = np.load('weights/weight-3.npy'))
		w4 = tf.Variable(initial_value = np.load('weights/weight-4.npy'))

	# the logits weights are always loaded (never pruned)
	logits_w = tf.Variable(initial_value = np.load('weights/weight-5.npy'))
	# forward pass of the network with loaded weights to determine accuracy
	correct_output = tf.placeholder(tf.int32, [None])
	inputs = tf.placeholder(tf.float32, [None, 784])

	layer1 = tf.nn.relu(tf.matmul(inputs, w1))
	layer2 = tf.nn.relu(tf.matmul(layer1, w2))
	layer3 = tf.nn.relu(tf.matmul(layer2, w3))
	layer4 = tf.nn.relu(tf.matmul(layer3, w4))
	logits = tf.nn.softmax(tf.matmul(layer4, logits_w))
	predictions = tf.cast(tf.math.argmax(logits, axis=1), tf.int32)

	acc, acc_percentage = tf.metrics.accuracy(labels=correct_output, predictions=predictions)
	# initialize global/local variables in TensorFlow graph
	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()
	tf.local_variables_initializer().run()
	# accuracy across the test set
	accuracy = sess.run(acc_percentage, feed_dict={inputs: x_test, correct_output: y_test})
	print('Accuracy: ', accuracy)

def main():
	#### TUNABLE PARAMETERS ####
	pruning_bool = True # boolean to detect whether or not we will prune
	pruning_type = 'unit' # type of pruning, valid entries: 'weight' and 'unit'
	k = 0.99 # percentage pruned, valid: 0.0 - 1.0
	
	if pruning_bool:
		weights_arr = [np.load('weights/weight-1.npy'), np.load('weights/weight-2.npy'), 
			np.load('weights/weight-3.npy'), np.load('weights/weight-4.npy')]
		pruned_weights = pruning_func(pruning_type, weights_arr, k)
		perform_forward_pass(pruning_bool, pruned_weights)
	else:
		perform_forward_pass(pruning_bool, [])

if __name__ == "__main__":
	main()
