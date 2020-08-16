# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 10:16:22 2017

@author: Mukund Bharadwaj
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
import itertools
import numpy as np
import theano
import theano.tensor as T
import lasagne
import time

def load_dataset(path, n_out):

    def load_images(filename):
        with open(path + filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8)
        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)
        data = data.reshape(-1, 1, 45,60)
        # The inputs come as bytes, we convert them to float32 in range [0,1].
        return data / np.float32(256)

    def load_labels(start=0,stop=50,repeat=1):
        data=np.arange(start,stop,1,np.int16)
        data=np.repeat(data,repeat)
        # The labels are vectors of integers now, that's exactly what we want.
        return data

    X_train = load_images('picdata_train.txt')
    y_train = load_labels(0,n_out,20)
    X_val=load_images('picdata_val.txt')
    y_val=load_labels(0,n_out,12)
    X_test=load_images('picdata_test.txt')
    y_test=load_labels(0,n_out,8)

    return X_train, y_train, X_val, y_val,X_test, y_test

def build_cnn(input_var, n_out):
    # As a third model, we'll create a CNN of two convolution + pooling stages
    # and a fully-connected hidden layer in front of the output layer.

    # Input layer, as usual:
    network = lasagne.layers.InputLayer(shape=(None, 1, 45,60),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=6, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=100,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network

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


def load_labels(start=0,stop=50,repeat=1):
    data=np.arange(start,stop,1,np.int16)
    data=np.repeat(data,repeat)
    # The labels are vectors of integers now, that's exactly what we want.
    return data

def load_data(acc):
    
    data = list()
    for maps in acc:
        for m in maps:
            data.append(np.ravel(m))
    return np.asarray(data)

def main(path, wts_path, n_out):
    
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(path, n_out)
    
    input_var = T.tensor4('inputs')
    
    # Create neural network model (depending on first command line parameter)
    network = build_cnn(input_var, n_out)
    
    #Load trained weights
    with np.load(wts_path + 'trained-CNN.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    
    #Getting Feature maps
    predict_fn = theano.function([input_var], lasagne.layers.get_output(
            lasagne.layers.get_all_layers(network)[1:-5], deterministic=True))
    
    def getFM(X,Y):
    
        net = list()
        for batch in iterate_minibatches(X, Y, 5, shuffle=False):
            inputs, targets = batch
            net.append(predict_fn(inputs))
        
        conv_1 = list()
        conv_2 = list()
        for n in net:
            conv_1.append(np.asarray(n[0]))
            conv_2.append(np.asarray(n[2]))
            
        return np.vstack(conv_1),np.vstack(conv_2)
    
    #Reading Training data
    conv_1,conv_2 = getFM(X_train, y_train)
        
    #Training RF
    clf1 = RandomForestClassifier(n_estimators=100)
    clf1.fit(load_data(conv_1), load_labels(0,n_out,120))
    
    clf2 = RandomForestClassifier(n_estimators=100)
    clf2.fit(load_data(conv_2), load_labels(0,n_out,640))
    
    #Reading Validation data
    conv_1,conv_2 = getFM(X_val, y_val)
    
    #Calibrating RF on calibrated data
    sig_clf1 = CalibratedClassifierCV(clf1, method="sigmoid", cv="prefit")
    sig_clf1.fit(load_data(conv_1), load_labels(0,n_out,72))
    
    sig_clf2 = CalibratedClassifierCV(clf2, method="sigmoid", cv="prefit")
    sig_clf2.fit(load_data(conv_2), load_labels(0,n_out,384))
    
    #Reading testing data
    conv_1,conv_2 = getFM(X_test, y_test)
    
    #Predicting Test values on RF
    pred1 = np.asarray(sig_clf1.predict_proba(load_data(conv_1)))
    pred1 = np.split(pred1,n_out)
    
    pred2 = np.asarray(sig_clf2.predict_proba(load_data(conv_2)))
    pred2 = np.split(pred2,n_out)
    
    out1 = list()
    out2 = list()
    for f,b in itertools.izip(pred1,pred2):
        out1.append(np.average(f, axis=0))
        out2.append(np.average(b, axis=0))
    
    out = list()
    for f,b in itertools.izip(out1,out2):
        lt = [(x*0.3+y*0.7) for x,y in zip(f,b)]
        out.append(lt)
    
    ans = list()
    for p in out:
        ans.append(np.argmax(p))
    acc_1,acc_2,acc_f = 0.0,0.0,0.0
    for i in xrange(n_out):
        if np.argmax(out[i])==i:
            acc_f += 1
        else:
            continue
        if np.argmax(out[i])==i:
            acc_1 += 1
        else:
            continue
        if np.argmax(out[i])==i:
            acc_2 += 1
        else:
            continue
    print 'Overall Accuracy: ' + str(acc_f*100/n_out)
    print 'FM1 Accuracy: ' + str(acc_1*100/n_out)
    print 'FM2 Accuracy: ' + str(acc_2*100/n_out)
    return ans