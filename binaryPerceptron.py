#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:07:09 2020
@author: Matthew Harding 201220275

"""
import numpy as np
#Open a file given a filename
def getFileData(fileName):  
    with open(fileName) as file:
        data = file.readlines()
        return data
        
#return a set of N randomised nums to inititialise weights (makes flexible)
def initWeights(n):
    return 0
        

#Init weights to 1
weights = [1,1,1,1]

#Set Learning Rate
lr = 0.01
#Set Training Iterations
ti = 20


#Data layout [Input, Input, Input, Input, Class]
def processData(data):
    processed = []
    for d in data:
        stripped = [x.strip() for x in d.split(',')]
        processed += [stripped]
    return processed


def train(weights, learningRate):
#   set up hyperparameters
    w = weights
    lr = learningRate
    bias = 1
#    get training data
    tData = processedData(getFileData("train.data"))
    
    for line in tData:
#        Check activation
#        Check class
#        Update Weights
        
    
    
model = train()