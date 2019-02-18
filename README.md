# Pruning Weights on a Large Neural Net

This is a Keras/TensorFlow implementation of MNIST classification with pruning. I've included two separate versions in two separate folder: small model and large model. The large model (which is specifically for the ForAI challenge) contains 4 fully-connected ReLU layers and 1 fully-connected softmax layer.

# How to use

If you would like to train the neural networks before pruning, run:

`python keras_model.py`

This will automatically train the model for 3 epochs (which you can modify) and store the weights in .npy format in the /weights folder. Note: biases are also stored, but for the purposes of this test, they are not used in the feedforward network to test the effects of pruning. Then, to check accuracy with pruning, run:

`python pruning_tf.py`

You can modify pruning values and pruning type (weight or unit pruning) at the bottom of the pruning_tf.py file.

The following are pruning level vs. accuracy on test set results for the large_model:

<img src="/img/weight_pruning.png" alt="Usage Data" width="500" height="400"/>
<img src="/img/unit_pruning.png" alt="Usage Data" width="500" height="400"/>

# Analysis
