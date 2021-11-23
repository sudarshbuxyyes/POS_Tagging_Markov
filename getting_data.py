import numpy as np
import math
import string
from collections import defaultdict
import pandas as pd
In [3]:
#Initialization

setPunctuation = set(string.punctuation)
suffixNoun = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
suffixVerb = ["ate", "ify", "ise", "ize"]
suffixAdjective = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
suffixAdverb = ["ward", "wards", "wise"]
In [21]:
# preprocessing datasets

def testingDataSet_preProcessing(vocabulary, file):
    preprossed = []
    
    with open(file, "r") as f:
        for index, word in enumerate(f):
            if not word.split():
                word = "--n--"
                preprossed.append(word)

            elif word.strip() not in vocabulary:
                word = tagAllotement(word)
                preprossed.append(word)

            else:
                preprossed.append(word.strip())

    return preprossed


def tagAllotement(word):
    if any(char.isdigit() for char in word):
        return "--unk_digit--"

    elif any(char in setPunctuation for char in word):
        return "--unk_punct--"

    elif any(char.isupper() for char in word):
        return "--unk_upper--"

    elif any(word.endswith(suffix) for suffix in suffixNoun):
        return "--unk_noun--"

    elif any(word.endswith(suffix) for suffix in suffixVerb):
        return "--unk_verb--"

    elif any(word.endswith(suffix) for suffix in suffixAdjective):
        return "--unk_adj--"

    elif any(word.endswith(suffix) for suffix in suffixAdverb):
        return "--unk_adv--"
    return "--unk--"

def preprocessWord(wordTagPair, vocabulary): 
    if not wordTagPair.split():
        word = "--n--"
        tag = "--s--"
        return word, tag
    else:
        word,tag = wordTagPair.split()
        if word not in vocabulary: 
            word = tagAllotement(word)
        return word, tag
    return None 
In [9]:
# datasets - training + testing
#trainingDataset - list holding the training data [format: word tag]
#voc - list holding the vocabulary [format: word] 
#vocabulary - dictionary [format: key:word, value:index]
vocabulary = {}  

with open("WSJ_02-21.pos", 'r') as file:
    trainingDataset = file.readlines()

with open("WSJ_24.pos", 'r') as file:
    testingDataset = file.readlines()
    
with open("hmm_vocab.txt", 'r') as file:
    voc = file.read().split('\n')

i = 0
for word in voc:
    vocabulary[word] = i
    i+=1
    
#preprocess testingDataSet
preprocessedTestingData = testingDataSet_preProcessing(vocabulary, "test.words") 
Part 1 - Parts-of-speech tagging
In [23]:
def calcCounts(trainingDataset, vocabulary):
    #countEmission - #(tag,word) pairs
    #countTransition - #(tag,tag) pairs
    #countTag - #tags
    countEmission = defaultdict(int)
    countTransition = defaultdict(int)
    countTag = defaultdict(int)
    
    previousTag = '--s--' 
    index = 0 
    
    for wordTagPair in trainingDataset:
        index+=1

        word, tag = preprocessWord(wordTagPair, vocabulary)
        
        countTransition[(previousTag, tag)] += 1
        countEmission[(tag, word)] += 1
        countTag[tag] += 1
        
        previousTag = tag
        
    return countEmission, countTransition, countTag
In [24]:
countEmission, countTransition, countTag = calcCounts(trainingDataset, vocabulary)
In [28]:
markovStates = sorted(countTag.keys())
print(markovStates)
['#', '$', "''", '(', ')', ',', '--s--', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
In [29]:
# accuracy of naive approach

def accuracyNaiveAppraoch(markovStates, preprocessedTestingData, testingDataset, vocabulary, countEmission):

    accuratePredictions = 0
    tagWordPair = set(countEmission.keys())

    for word, testingData_wordTag in zip(preprocessedTestingData, testingDataset): 

        testingData_wordTag = testingData_wordTag.split()
        if len(testingData_wordTag) == 2:
            correctTag = testingData_wordTag[1]
        else:
            continue

        maximumCount = 0
        maximizingPOS = ''
    
        if word not in vocabulary:
            continue
        elif word in vocabulary:
            for possibleTag in markovStates:
                possibleKey = (possibleTag, word)
                if possibleKey in countEmission:
                    count = countEmission[possibleKey]
                    if count > maximumCount:
                        maximumCount = count
                        maximizingPOS = possibleTag
                        
            if maximizingPOS == correctTag:
                accuratePredictions += 1

    accuratePredictions /= len(testingDataset)
    return accuratePredictions
In [34]:
accuracyNaive = accuracyNaiveAppraoch(markovStates, preprocessedTestingData, testingDataset, vocabulary, countEmission)
print("Naive accuracy is "+str(accuracy_predict_pos*100)+" %")
Naive accuracy is 88.88563993099213 %
Part 2 - Hidden Markov Models for POS
In [37]:
def calcTransitionMatrix(countTransition, countTag, epsilon, markovStates):
    
    numberStates = len(markovStates)
    transitionMatrix = np.zeros((numberStates, numberStates))
    tagTagPair = set(countTransition.keys())

    for i in range(numberStates):
        for j in range(numberStates):
            countNextTag = 0
            key = (markovStates[i],markovStates[j])
            
            if key in countTransition: 
                countNextTag = countTransition[key]
                
            countPreviousTag = countTag[key[0]]
            transitionMatrix[i][j] = (countNextTag+epsilon)/(countPreviousTag+(numberStates*epsilon))

    return transitionMatrix
In [41]:
epsilon = 0.001
transitionMatrix = calcTransitionMatrix(countTransition, countTag, epsilon, markovStates)
In [53]:
def calcEmissionMatrix(countEmission, countTag, epsilon, markovStates, vocabulary):

    numberStates = len(markovStates)
    numberWords = len(vocabulary)
    emissionMatrix = np.zeros((numberStates, numberWords))
    tagWordPair = set(list(countEmission.keys()))
    
    for i in range(numberStates):
        for j in range(numberWords):
            countWord = 0
            key =  (markovStates[i],vocabulary[j])

            if key in countEmission: 
                countWord = countEmission[key]

            countTags = countTag[key[0]]

            emissionMatrix[i][j] = (countWord+epsilon)/(countTags+(epsilon*numberWords))

    return emissionMatrix
In [57]:
emissionMatrix = calcEmissionMatrix(countEmission, countTag, epsilon, markovStates,  list(vocabulary))