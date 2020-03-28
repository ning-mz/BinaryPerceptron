#
#Student: 
#ID: 

import numpy as np
np.seterr(all='ignore')

#define the path to data file
trainDataFilePath = 'train.data'
testDataFilePath = 'test.data'

#set hyperparameters of perceptron
Epochs = 20
LearningRate = 0.1
L2Reg = 0

#define a class of a perceptron and implement it for question 3
class Perceptron:
    def __init__(self, trainData, testData, epochs, learningRate, L2Reg):
        self.trainAccuracy = 0
        self.testAccuracy = 0
        
        #define perceptron related parameters
        self.dataDimension = np.size(trainData, 1) - 1
        self.weights = np.zeros(self.dataDimension)
        self.trainData = trainData
        self.testData = testData
        self.epochs = epochs
        self.learningRate = learningRate
        self.L2Reg = L2Reg

    #training the percetpron and update weights for self perceptron
    def trainPerceptron(self):
        prediction = np.zeros(self.trainData.shape[0])
        label = self.trainData[:, 5]
        sampleCounter = 0
        wrongCounter = 0
        for t in range(0, self.epochs):
            # print(self.weights)
            for index in range(0, self.trainData.shape[0]):
                result = np.dot(self.trainData[index][:5], self.weights)

                #sigmoid activation funciton for predictions
                sigmoid = 1/(1+np.exp(-result))
                if sigmoid > 0.7:
                    prediction[index] = 1.0
                else: 
                    prediction[index] = -1.0

                if label[index] != 0:   
                    sampleCounter = sampleCounter + 1             
                    update = (label[index]-prediction[index])*self.learningRate*self.trainData[index][:5]

                    #if l2 regularization is applied
                    if self.L2Reg != 0:
                       update = update - (2*self.L2Reg*self.weights)   

                    #print(update)      
                    self.weights += update     

                    #count the wrong predict numbers to get accuracy
                    if (label[index]-prediction[index]) != 0:
                        wrongCounter = wrongCounter + 1

                #calculate error and update weight
        print('Overall Training accuracy: {:.2%}'.format(1 - (wrongCounter / sampleCounter)))
        #print('Weights: {}'.format(self.weights))
   
    #test the perceptron
    def testPerceptron(self):    
        label = self.testData[:, 5]
        wrongCounter = 0
        counter = 0
        for index in range(0, self.testData.shape[0]):
            result = np.dot(self.testData[index][0:5], self.weights)

            #sigmoid activation funciton
            sigmoid = 1/(1+np.exp(-result))
            if sigmoid > 0.7:
                result = 1.0
            else: 
                result = -1.0

            if label[index] != 0:
                counter = counter + 1
                if label[index] != result:
                    wrongCounter = wrongCounter + 1

        self.testAccuracy = 1.0 - (wrongCounter / counter)
        print('Test finished, accuracy: {:.2%}'.format(self.testAccuracy))    

#preprocess the dataset for help training
def dataPreProcess(dataFilePath, labelSet):
    dataRaw = np.genfromtxt(dataFilePath, delimiter=',', usecols = (0,1,2,3))
    labelRaw = np.genfromtxt(dataFilePath , delimiter=',', usecols = (4), dtype=np.object)
    
    #labelset will detemine the positive and negative label as 1/-1
    #bias input should be 0 or 1???
    bias = np.ones(dataRaw.shape[0])        
    labelProcessed = np.zeros(labelRaw.shape[0])

    for i in range(0, labelRaw.shape[0]):
        if labelRaw[i] == b'class-1':
            labelProcessed[i] = labelSet[0]
        elif labelRaw[i] == b'class-2':
            labelProcessed[i] = labelSet[1]
        elif labelRaw[i] == b'class-3':
            labelProcessed[i] = labelSet[2]
    #combine data together
    processedData = np.c_[bias, dataRaw, labelProcessed]
    #shuffle the data 
    np.random.shuffle(processedData)
    return processedData

#load dataset
trainDataFile = np.genfromtxt(trainDataFilePath, delimiter=',', usecols = (0,1,2,3))
testDataFile = np.genfromtxt(testDataFilePath, delimiter=',', usecols = (0,1,2,3))

#determine the case of dataset for training
labelSet = np.array([[1, -1, 0], [1, 0, -1], [0, 1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
cases = ["1 vs 2", "1 vs 3", "2 vs 3", "1 vs rest", "2 vs rest","3 vs rest"]
print('Perceptron algorithm for COMP337 Ass1')
print('Maizhen Ning     ID:201376369')
#print(dataPreProcess(trainDataFilePath, labelSet[0]))

#start the question 4 & 6
print('*********************** Start question 4 & 6 **************************')
print('-------------------------')
for t in range(0, np.size(labelSet, 0)):
    print('Running: ',cases[t])
    perceptron = Perceptron(dataPreProcess(trainDataFilePath, labelSet[t]), 
                        dataPreProcess(testDataFilePath, labelSet[t]),
                        Epochs, LearningRate, L2Reg)
    perceptron.trainPerceptron()     
    perceptron.testPerceptron()
    print('-------------------------')

#start the question 7 of assignment
print('*********************** Start question 7 **************************')
L2Reg = 0.01

for time in range(0, 5):
    print('--------------L2 regularisation is: ', L2Reg, '--------------')
    for t in range(0, np.size(labelSet, 0)):
        print('Running: ',cases[t])   
        perceptron = Perceptron(dataPreProcess(trainDataFilePath, labelSet[t]), 
                        dataPreProcess(testDataFilePath, labelSet[t]),
                        Epochs, LearningRate, L2Reg)
        perceptron.trainPerceptron()     
        perceptron.testPerceptron()
        print('-------------------------')
    L2Reg = L2Reg * 10
    
