# Pruning Weights on a Large Neural Net

This is a Keras/TensorFlow implementation of MNIST classification with pruning. This model (which is specifically for the ForAI challenge) contains 4 fully-connected ReLU layers and 1 fully-connected softmax layer.

# Installation Requirements
To run the model, you will need to install Tensorflow and Keras. You can install both using pip. To speed up training time, you can install Tensorflow-gpu, though I have already trained weights for use in the pruning experiments.

# How to use

If you would like to train the neural networks before pruning, run:

`python keras_model.py`

This will automatically train the model for 3 epochs (which you can modify) and store the weights in .npy format in the /weights folder. Note: biases are also stored, but for the purposes of this test, they are not used in the feedforward network to test the effects of pruning. 
Weights have already been trained and stored in the weights/ folder, so you don't need to train the model again. To check accuracy with pruning, run:

`python pruning_tf.py`

You can modify pruning values and pruning type (weight or unit pruning) at the bottom of the pruning_tf.py file.

The following are graphs depict the effects of k (pruning level) on the accuracy of the model on the MNIST test set:

<img src="/img/weight_pruning.png" alt="Usage Data" width="500" height="400"/>
<img src="/img/unit_pruning.png" alt="Usage Data" width="500" height="400"/>

# Analysis

From the graphs, it's clear to see even pruning a lot of weights doesn't result in a drastic loss in accuracy. In the case of weight pruning, you would need to prune at least 95% of weights before noticing any noticeable performance decrease. Why is this?

First, it’s important to look at the dataset. MNIST is a dataset of handwritten digits fit into a 28 by 28 grid. However, only a small set of pixel values encode actual digit markings (black pixels). The rest of the pixel values encode white (or empty space). In the first image of the training set, about 70% of the pixel values were 0. This means that the majority of weights in the first layers have no impact, as they are multiplied by 0. This value of 0 then propagates down the feedforward network to the output. Therefore, deleting the weights that correspond to connections of 0 in the input will have no impact on the result. Considering that a large percentage of input corresponds to this 0 value, it is probable that a weight pruned will not be useful to the output.

Another reason is the very of nature pruning: it eliminates weights/columns of lower magnitude, and thus weights/columns with less impact on the output. Since we prune weights of lower importance first, we can sustain performance to a certain degree. 

It’s also clear to see unit pruning performed worse than weight pruning. Why? Weight pruning removes the salience of features, while unit pruning removes features (or neurons)  entirely. This causes issues in the case of unit pruning because if one weight in a column vector is large (indicating salience) but the remaining weights are small, this entire neuron could be eliminated. The correct approach is to reduce the impact of the other weights and thus reduce quality a little instead of entirely removing a feature. This is exactly weight pruning, which looks at individual weights, and eliminates them only if they have limited impact on the output. This is a more case-by-case approach to increasing efficiency while maintaining quality.

One interesting result I found was the performance of weight pruning actually jumped by about 0.5% (still an improvement) between k = 0 and k = 20. This is likely because the Keras model produced biases in addition to weights by default, and in my feedforward network, I only used the weights (as per instructions).


