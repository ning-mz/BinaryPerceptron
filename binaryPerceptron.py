#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:07:09 2020
@author: Matthew Harding 201220275

"""
import numpy as np
import random

#Open a file given a filename
def getFileData(fileName):  
    with open(fileName) as file:
        data = file.readlines()
        return data

#return a set of N randomised nums to inititialise weights (makes flexible)
def initWeights(n):
    return 0
       

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

# make a prediction, returns 1 or 0
def predict(row,weights):
    # set weighted sum as bias weight to start for each row
    weightedSum = weights[0]
    for i in range(len(row)-1):
        weightedSum += weights[i+1] * float(row[i])
    return 1.0 if weightedSum >= 1.0 else -1.0    

# given a scenario(a,b,c), remove unnecessary data  
def splitData(scenario, data):
    scn = scenario
    # print(scn, "\n")
    newdata=[]
    if scn == "a":
        for row in data:
            if row[4] =='class-1':
                row[4] = 1.0
                newdata+=[row]
            elif row[4] == 'class-2':
                row[4] = -1.0
                newdata+= [row]
        return newdata
    
    elif scn == "b":
        for row in data:
            if row[4] == 'class-2':
                row[4] = 1.0
                newdata+=[row]
            elif row[4] == 'class-3':
                row[4] = -1.0
                newdata+=[row]
        return newdata
    
    elif scn == "c":
        for row in data:
            if row[4] =='class-1':
                row[4] = -1.0
                newdata+=[row]
            elif row[4] == 'class-3':
                row[4] = 1.0
                newdata+=[row]
        return newdata
    
    
def train(learningRate,epochs,scenario):
#   set up hyperparameters
    scn = scenario
    w = [0.1,0.1,0.1,0.1]
    lr = learningRate
    bias = 0.1
    e = epochs
    iteration = 0
    
#   get training data and remove uneccesary classes, shuffle
    tData = processData(getFileData("train.data"))
    tData = splitData(scn, tData)
    random.shuffle(tData)
    
#   insert bias such that weights = [bias, w1, w2, w3, w4]
    w.insert(0,bias)

#   While iterations < number of epochs
    while iteration < e:
        # sum errors for each epoch 
        numOfErrors = 0
        # for each row in training data make a prediction 
        # and update weights if there is an error 
        for line in tData:
            prediction = predict(line, w)
            error = line[len(line)-1] - prediction
            if error!= 0:
                numOfErrors +=1
            # update bias
            w[0] = w[0] + (lr * error)
            # update by adding learning rate*error*input to existing weights
            for i in range(len(line)-1):
                w[i+1]=w[i+1] +  lr * error * float(line[i])
        # print(numOfErrors, "prediction errors in epoch ", iteration)
        iteration += 1
    return w 
        
    
def test(modelWeights, scenario):
    testData = processData(getFileData("test.data"))
    testData = splitData(scenario, testData)
    random.shuffle(testData)
    predictions = []
    actual = []
    for row in testData:
        p = predict(row, modelWeights)
        predictions.append(p)
        actual.append(row[4])
    accuracy= calculateAccuracy(predictions, actual)
    # print("predictions",predictions, "\n")
    # print("actual",actual,"\n")
    print("Test dataset accuracy for scenario" ,scenario," is ",accuracy,"%")
    return predictions

def calculateAccuracy(predictions, actual):
    positives = 0
    for i in range(len(predictions)):
        if predictions[i] == actual[i]:
            positives += 1
    return(positives/float(len(predictions)))*100
        
modela = train(lr, ti,"a")
modelb = train(lr,ti,"b")
modelc = train(lr,ti,"c")

test(modela, "a")
test(modelb, "b")
test(modelc, "c")

# 1 vs rest, take 3 classifiers and return maxprobability

def oneVsRest():
    testData = processData(getFileData("test.data"))
    random.shuffle(testData)
    predictions=[]
    actual=[]
    for row in testData: 
        class1 = predict(row,modela)
        class2 = predict(row,modelb)
        if class1 == 1.0:
            predictions.append('class-1')
        elif class2 == 1.0:
            predictions.append('class-2')
        else:
            predictions.append('class-3')
        actual.append(row[4])
    print("one vs many accuracy is" , calculateAccuracy(predictions,actual))
    
    
oneVsRest()
    
            