# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 15:23:12 2015

@author: ankit
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

"""
##################################
#                                #
#           Set flags            #
#                                #
##################################
"""

#allow bias units in implemetation
bias = False

#rate decay at 20%, 40%, 60% and 80% of training by factorConv and factorFull
rateDecay = False
factorConv = 10
factorFull = 10

#initial learning rates
rateConv = 0.001
rateFull = 0.001

#iterations on training
iterations = 1500

"""
##############################################
#                                            #
#   Fully connected layers architecture      #
#                                            #
##############################################
"""

neurons = [392, 80, 20, 4]

"""
    modify neurons based on flag
"""
if bias:
    for i in xrange(len(neurons)-1):
        if i!=0:
            neurons[i]+=1
 
"""
    initialize errors, dels, synapse matrix
"""    
nlayers = len(neurons)

layers = []
synapse = []

#holds weights to be updated after backprop on whole dataset
tempSynapse = []

for i in xrange(len(neurons)):
    layers.append(np.zeros(neurons[i]))
    if i!=0:
        synapse.append(((np.random.rand(neurons[i], neurons[i-1])*2)-1)*0.1)
        tempSynapse.append(np.zeros((neurons[i], neurons[i-1])))


dels = [0 for i in range(nlayers-1)]
error = np.zeros(neurons[nlayers-1])


"""
##################################
#                                #
#   Convolution layers filters   #
#                                #
##################################
"""

#number of filters
nFilters = 8

#filter dimensions
filterSize = 7

filters = []

#holds weights to be updated after backprop on whole dataset
tempFilters = []

for i in xrange(nFilters):
    filters.append(((np.random.rand(filterSize, filterSize)*2)-1)*0.1)
    tempFilters.append(np.zeros((filterSize, filterSize)))

#contains feature maps after convolution on image
featureMaps = []

#processed image data
imgFeats = []

#contains features after average pooling
postpool = []


"""
#############################
#                           #
#   Convolutional layers    #
#                           #
#############################
"""


"""
    convForwardPass(X)
    
    Parameters
    X: features from dataset
    
    Returns
    vector of features for fully connected network
    
    X ----> [CONV] ----> [AVG POOLING] ----> [SIGMOID]
"""

def convForwardPass(X):
    global nFilters, filters, featureMaps, postpool
    featureMaps = []
    postpool = []
    X = X.reshape(X.size**0.5, X.size**0.5)
    for i in xrange(nFilters):
        featureMaps.append(signal.convolve2d(X, filters[i], 'valid'))
        prepool = featureMaps[i].reshape(featureMaps[i].shape[0]/2, 2, featureMaps[i].shape[1]/2, 2)
        postpool.append(prepool.mean(axis=(1,3)))
        
    return sigmoid(np.hstack(np.hstack(postpool)))
    
"""
    convBackProp(X)
    
    updates tempFilters
    vector of features for fully connected network
    FC dels ----> CONV dels ----> FILTER dels ----> [UPSCALE] ----> [CROSS-CORRELATION]
"""
def convBackProp():
    global nFilters, filters, featureMaps, synapse, layers, dels, rateConv, tempFilters, imgFeats
    
    dim = imgFeats.shape[0]**0.5    
    im = imgFeats.reshape(dim, dim)
    convDel = sigmoid(layers[0], True)*np.dot(synapse[0].T, dels[0])
    
    filterDels = np.split(convDel, nFilters)
    i = 0
    for filterDel in filterDels:
        
        dim = filterDel.shape[0]**0.5
        filterDel = filterDel.reshape(dim, dim)
        filterDel = np.kron(np.array(filterDel), np.ones((2,2)))
        tempFilters[i] += rateConv*signal.convolve2d(im, np.rot90(filterDel, 2), 'valid')
        i = i + 1

"""
##################################
#                                #
#   Fully connected layers       #
#                                #
##################################
"""

"""
    sigmoid(z, derivative = False)
    
    Parameters:
    z: argument to apply sigmoid on
    derivative: True if derivative of sigmoid to be returned
    
    Returns:
    if derivative = False, then sigmoid(z)
    if derivative = True, then sigmoid'(z)
"""
def sigmoid(z, derivative = False):
    if derivative:
        return z*(1-z)
    else:
        return 1./(1 + np.exp(-z))


"""
    train(data, test)
    
    Parameters:
    data: data to be trained on
    test: data for testing
    
    each datum is of the format [X, y]
    where,
        X is the feature list, and
        y is the target label
"""
def train(data, test):
    global iterations, neurons, layers, synapse, error, dels, rateConv, rateFull, tempFilters, tempSynapse, imgFeats, iterations
    
    ploterror = np.zeros(iterations)
    plottesterror = np.zeros(iterations)
    
    for iteration in xrange(iterations):
        
        """rate decay updates"""
        if iteration in [0.20*iterations, 0.4*iterations, 0.6*iterations, 0.8*iterations] and rateDecay:
            print iteration, ': completed\n'
            rateConv = rateConv / 10
            rateFull = rateFull / 10
        
        
        sumerror, testerror = 0, 0
        
        for index in xrange(len(data)):
            imgFeats = np.array(data[index][0])
            X = convForwardPass(imgFeats)
            
            if bias:            
                X = np.insert(X, 0, 1)
            
            y = data[index][1]
            layers = forwardPass(layers, synapse, X)
            
            error = y - layers[nlayers-1]
            sumerror = sumerror + sum(abs(error))
            
            
            for i in xrange(nlayers-2, -1, -1):
                if i==nlayers-2:
                    dels[i] = sigmoid(layers[i+1], True)*error
                else:
                    dels[i] = sigmoid(layers[i+1], True)*np.dot(synapse[i+1].T, dels[i+1])
            
            ineurons = nlayers-1
            for i in xrange(nlayers-2, -1, -1):
                """update dot product"""
                tempSynapse[i] += rateFull*np.dot(dels[i].reshape(neurons[ineurons], 1), layers[i].reshape(1, neurons[ineurons-1]))
                if i == 0:
                    convBackProp()
                ineurons = ineurons - 1
        
        #update weights after back prop on all training examples
        for index in xrange(len(synapse)):
            synapse[index]+=tempSynapse[index]
            tempSynapse[index] = tempSynapse[index]-tempSynapse[index]
        
        for index in xrange(len(filters)):
            filters[index]+=tempFilters[index]
            tempFilters[index] = tempFilters[index]-tempFilters[index]
        
        
        #calculate error
        for index in xrange(len(test)):
            X = convForwardPass(np.array(data[index][0]))
            
            if bias:
                X = np.insert(X, 0, 1)
            
            y = data[index][1]
            
            layers = forwardPass(layers, synapse, X)
            
            tt = np.zeros(neurons[nlayers-1])
            tt[np.argmax(layers[nlayers-1])] = 1
                
            testerror += sum(abs(y - tt))
            
        
        #store average error for plotting purposes
        ploterror[iteration] = sumerror/len(data)
        plottesterror[iteration] = testerror/len(test)
        
        #display error every 100 iterations
        if iteration%100 == 0:
            print 'Iterations:', iteration, ' Error:', sumerror/len(data), ' Test Error:', testerror/len(test)
            
    #plot the train and test errors
    plotx = range(iterations)
    plt.figure(1)
    plt.plot(plotx, ploterror)
    plt.plot(plotx, plottesterror)
    plt.show()


"""
    forwardPass(layer, synapse, X)
    Parameters:
    layer
    synapse
    X
"""
def forwardPass(layer, synapse, X):
    layer[0] = X
    for weights in xrange(len(synapse)):
        if weights == 0:
            layer[1] = sigmoid(np.dot(synapse[0], layer[0]))
            if bias:
                layer[1] = np.insert(layer[1], 0, 1)
        else:
            layer[weights + 1] = sigmoid(np.dot(synapse[weights], layer[weights]))
            if bias and weights!=len(synapse):
                layer[weights + 1] = np.insert(layer[weights + 1], 0, 1)
    return layer
            

def predict(X):
    global synapse
    output = 0
    for weights in xrange(len(synapse)):
        if weights == 0:
            output = sigmoid(np.dot(synapse[0], X))
        else:
            output = sigmoid(np.dot(synapse[weights], output))


    

"""
#############################
#                           #
#         Training          #
#                           #
#############################
"""

import dataset

data = dataset.data
test = dataset.test

train(data, test)




