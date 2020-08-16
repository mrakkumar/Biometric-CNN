#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import time

import numpy as np
import theano
import theano.tensor as T

from lasagne import layers
import lasagne.nonlinearities
from nolearn.lasagne import BatchIterator
from nolearn.lasagne import NeuralNet
from skimage import transform as tf

# ################## Prepare the hand vein dataset ##################

def load_dataset():

    def load_images(filename):
        with open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 45,60)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        return data / np.float32(256)

    def load_labels(start=0,stop=50,repeat=1):
        data=np.arange(start,stop,1,np.int8)
        data=np.repeat(data,repeat)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    X_train = load_images('picdata_train.txt')
    y_train = load_labels(0,5,7)
    X_val=load_images('picdata_val.txt')
    y_val=load_labels(0,5,2)

    return X_train, y_train, X_val, y_val

# ##################### Augmentation ###########################

rots = np.asarray((-20,-10,-5,5,10,20)) / (360 / (2.0 * np.pi))
dists = (-1,1)

def manipulateTrainingData(Xb):
    retX = np.zeros((Xb.shape[0], Xb.shape[1], Xb.shape[2], Xb.shape[3]), dtype='float32')
    for i in range(len(Xb)):
        dist = dists[np.random.randint(0, len(dists))]
        rot = rots[np.random.randint(0, len(rots))]
        scale = np.random.uniform(0.9,1.10)
        tform = tf.SimilarityTransform(rotation=rot, translation=dist, scale=scale)
        retX[i,0,:,:] = 256.0 * tf.warp(Xb[i,0,:,:]/256.0,tform) # "Float Images" are only allowed to have values between -1 and 1
    return retX


# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_cnn():
    return NeuralNet(
        # Geometry of the network
        layers=[
            ('input', layers.InputLayer),
            ('conv1', layers.Conv2DLayer),
            ('pool1', layers.MaxPool2DLayer),
            ('conv2', layers.Conv2DLayer),
            ('pool2', layers.MaxPool2DLayer),
            ('hidden4', layers.DenseLayer),
            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 1, 45,60), #None in the first axis indicates that the batch size can be set later
        conv1_num_filters=6, conv1_filter_size=(5, 5), conv1_nonlinearity = lasagne.nonlinearities.rectify, conv1_W=lasagne.init.GlorotUniform(), pool1_pool_size=(2, 2), #pool_size used to be called ds in old versions of lasagne
        conv2_num_filters=32, conv2_filter_size=(5, 5), conv2_nonlinearity = lasagne.nonlinearities.rectify, pool2_pool_size=(2, 2), #pool_size used to be called ds in old versions of lasagne
        hidden4_num_units=100, hidden4_nonlinearity=lasagne.nonlinearities.rectify,
        output_num_units=5, output_nonlinearity=lasagne.nonlinearities.softmax, 

        # learning rate parameters
        update_learning_rate=0.01,
        update_momentum=0.5,
        regression=False,
        # We only train for 10 epochs
        max_epochs=100,
        verbose=1,

        # Training test-set split
        eval_size = 0.2
        )


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='cnn', num_epochs=5):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_dataset()
    X_train = manipulateTrainingData(X_train)

    print (len(X_train))
    print (len(X_val))
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn()
    d = network.fit(X_train, y_train);

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(d)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(d, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.5)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(d, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 5, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 5, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

    np.savez('trained-CNN.npz', *lasagne.layers.get_all_param_values(network))
main('cnn',100)