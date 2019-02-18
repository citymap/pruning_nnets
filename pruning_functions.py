import numpy as np
import math
import sys

'''
This files contains both weight and unit pruning methods.
'''

# Prune an individual weight using the weight pruning technique.
# I converted all weights to absolute value format, sorted them, 
# found the pivot point, and set values below this pivot point to 0.
def individual_weight_pruning(weight, k):
	flattened_arr = np.absolute(np.ndarray.flatten(weight))
	length = len(flattened_arr)
	sorted_arr = np.sort(flattened_arr)
	pivot = sorted_arr[int(k * length)]
	for row in range(weight.shape[0]):
		for col in range(weight.shape[1]):
			if (abs(weight[row][col]) < pivot):
				weight[row][col] = 0
	return(weight)

# Prune an individual weight using the unit pruning technique.
# I squared all weights, find the column sums, and sort these column
# sums. I then sort these column sum, find the pivot point, and set columns
# with squared sum less than this pivot to 0.
def individual_unit_pruning(weight, k):
	squared = np.square(np.array(weight))
	columnwise = np.sum(squared, axis=0)
	length = len(columnwise)
	sorted_columnwise = np.sort(columnwise)
	pivot = sorted_columnwise[int(k * length)]
	for col in range(length):
		if columnwise[col] < pivot:
			weight[:,col] = np.zeros(weight.shape[0])
	return(weight)

# return an array of weight prunned weights
def weight_pruning(weights_arr, k):
	pruned_weights = []
	for weight in weights_arr:
		pruned_weight = individual_weight_pruning(weight, k)
		pruned_weights.append(pruned_weight)
	return(pruned_weights)

# return an array of unit pruned weights
def unit_pruning(weights_arr, k):
	pruned_weights = []
	for weight in weights_arr:
		pruned_weight = individual_unit_pruning(weight, k)
		pruned_weights.append(pruned_weight)
	return(pruned_weights)

def pruning_func(pruning_type, weights_arr, k):
	if (pruning_type == 'weight'):
		return(weight_pruning(weights_arr, k))
	elif (pruning_type == 'unit'):
		return(unit_pruning(weights_arr, k))
	else:
		print('Invalid pruning method. Exiting program')
		sys.exit(1)
