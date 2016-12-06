import os
import sys
import pandas
import random
import itertools

import numpy as np
from scipy import spatial


INPUT_FILE_NAME = "TenDigitPatterns.txt"
INPUT_PATTERN = '#'
PATTERN_ROW_LEN = 5
PATTERN_VECTOR_LEN = 35

THRESHOLD = -1 # Threshold used at output layer

def readInputFile(fileName):
    filHandle = open(fileName)
    return map(lambda pattern:pattern[:PATTERN_VECTOR_LEN], filHandle.readlines())

def getCosineSimilarity(vec1, vec2): # Length of both vectors is assumed to be same.
    return round(1 - spatial.distance.cosine(vec1, vec2), 2)

def constructOrthogonalMatrix(patterns):
    mat = np.zeros(shape = (len(patterns), len(patterns)))
    counter = 0
    for outerPattern in patterns:
        l = []
        for innerPattern in patterns:
            l.append(getCosineSimilarity(innerPattern, outerPattern))
        mat[counter] = l
        counter = counter + 1
    return mat

def createVizForPatterns(pattern):
    lolFunc = lambda lst, sz: [lst[i:i + sz] for i in range(0, len(lst), sz)] # lambda function
    lol = lolFunc(pattern, PATTERN_ROW_LEN)

    for row in lol:
        rowToDisplay = map(lambda ele: '.' if ele == 1 else ' ', row)
        if(rowToDisplay != None):
            for e in rowToDisplay:
                print e,
            print "\n"

def induceNoiseInPattern(pattern, numOfBitsToChange, isError):
    changedPattern = list(pattern) # dont change the original list
    randomIndices = random.sample(range(0, 34), numOfBitsToChange)
    for ran in randomIndices:
        if(isError):
            changedPattern[ran] = -changedPattern[ran]
        else:
            changedPattern[ran] = 0
    return changedPattern

def checkIfPatternIsSame(patternToTest, basePattern):
    if(type(patternToTest[0]) != type(1.0)):
        patternToTest = map(lambda x:x.item(), patternToTest)

    if (len(patternToTest) == len(basePattern)):
        uncommon = [1 for x, y in zip(patternToTest, basePattern) if x != y]
        if (len(uncommon) == 0): return True
    return False

def getNumericNumForPattern(pattern, patterns): # assuming PATTERNS Vector is already filled
    match = -1
    for patternNum in range(0, len(patterns)):
        if(checkIfPatternIsSame(pattern, patterns[patternNum])): return patternNum
    return match
    

def convertPatternToVector(pattern):
    return np.array(map(lambda x: 1 if x == INPUT_PATTERN else -1, pattern)) # using bipolar vectors

def convertPatternsToInputVectors(patterns):
    inputMatrix = np.zeros(shape = (len(patterns), PATTERN_VECTOR_LEN))
    for i in range(0, len(patterns)):
        inputMatrix[i] = convertPatternToVector(patterns[i]) # convert '#' pattern to 1 and -1
    return inputMatrix

'''
    Activation Function
    f(x) = {
            +1 if x > THRESHOLD
            0 if -THRESHOLD <= x <= THRESHOLD
            -1 if x < -THRESHOLD
            }
'''
def runOutputLayerTransferFunc(outputFromNet):
    out = []
    for i in range(0, len(outputFromNet)):
        if(outputFromNet[i] > THRESHOLD): out.append(1.0)
        elif(outputFromNet[i] < -THRESHOLD): out.append(-1.0)
        else: out.append(0)
    return out

def getOutputFromLayer(weightMatrix, inputPattern):
    return weightMatrix.dot(inputPattern)

def computeWeightMatrixUsingHebbRule(patterns):
    weightMatrix = np.zeros(shape = (PATTERN_VECTOR_LEN, PATTERN_VECTOR_LEN))
    for pattern in patterns:
        weightMatrix += map(lambda x: pattern if x == 1 else -pattern, pattern)

    return weightMatrix        

def main():
    numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    patterns = readInputFile(INPUT_FILE_NAME) # Read the input patterns file
    patterns =  convertPatternsToInputVectors(patterns) # convert patterns to input-matrix

    ''' Cosine Similarity Matrix
    print pandas.DataFrame(constructOrthogonalMatrix(patterns), numbers, numbers)
    '''

    '''
     Part(a)
     Start with storing all patterns and checking how many can be recalled.
     Then start removing one pattern and check again. Catch here is that when
     removing a pattern we need to consider all combinations. So if removing one
     pattern then (10C9) combinations should be considered.
     '''
    results = []
    for run in range(10, 0, -1):
        combinations = itertools.combinations(numbers, run)
        for combination in combinations:
            inputPatterns = map(lambda x:patterns[x], combination)
            weightMatrix1 = computeWeightMatrixUsingHebbRule(inputPatterns)
            output = map(lambda pattern:getNumericNumForPattern(runOutputLayerTransferFunc(
                getOutputFromLayer(weightMatrix1, pattern)), patterns), patterns)
            output = set(output)
            finalOutput = [x for x in output if x != -1]
            finalOutput.sort()
            results.append((run, combination, finalOutput))

    for run in range(10, 0, -1):
        subsetOfResults = [x for x in results if x[0] == run]
        maxResult = max(subsetOfResults, key=lambda x: len(x[2]))
        maxResults = [i for i in subsetOfResults if len(i[2]) == len(maxResult[2])]

        print len(maxResults)
        for res in maxResults:
            print res


    '''
    Part(b)
     Analysis of how much noise can the network handle. Optimum Configuration for training = [1, 2, 4, 5, 7, 8]
     We go systematically here, starting with just one bit error/missing value in patterns. Error can be while
     training as well as testing. So we analyse both of these scenarios.

    optimalTraining1 = [patterns[1], patterns[2], patterns[4], patterns[5], patterns[7], patterns[8]]
    weightMatrix1 = computeWeightMatrixUsingHebbRule(optimalTraining1)

    print "Below is for First Optimal Train Patterns"
    for bitsToChange in range(0, 35, 1):
        randomPatternToTest = random.choice(optimalTraining1)
        erroredPattern = induceNoiseInPattern(randomPatternToTest, bitsToChange, True)
        missingValsPattern = induceNoiseInPattern(randomPatternToTest, bitsToChange, False)

        outputWithErroredPattern = getNumericNumForPattern(runOutputLayerTransferFunc(
            getOutputFromLayer(weightMatrix1, erroredPattern)), patterns)
        outputWithMissingValsPattern = getNumericNumForPattern(runOutputLayerTransferFunc(
            getOutputFromLayer(weightMatrix1, missingValsPattern)), patterns)

        print "Total Bits Changed =",
        print bitsToChange,
        print "Random Pattern =",
        print getNumericNumForPattern(randomPatternToTest, patterns),
        print " ",
        print "Output After Error Induced =",
        print outputWithErroredPattern,
        print " ",
        print "Output After Missing Vals Inserted =",
        print outputWithMissingValsPattern,
        print "\n"

    optimalTraining2 = [patterns[0], patterns[1], patterns[2], patterns[4], patterns[6], patterns[7]]
    weightMatrix2 = computeWeightMatrixUsingHebbRule(optimalTraining2)

    print "Below is for Second Optimal Train Patterns"
    for bitsToChange in range(0, 35, 1):
        randomPatternToTest = random.choice(optimalTraining2)
        erroredPattern = induceNoiseInPattern(randomPatternToTest, bitsToChange, True)
        missingValsPattern = induceNoiseInPattern(randomPatternToTest, bitsToChange, False)

        outputWithErroredPattern = getNumericNumForPattern(runOutputLayerTransferFunc(
            getOutputFromLayer(weightMatrix2, erroredPattern)), patterns)
        outputWithMissingValsPattern = getNumericNumForPattern(runOutputLayerTransferFunc(
            getOutputFromLayer(weightMatrix2, missingValsPattern)), patterns)

        print "Total Bits Changed =",
        print bitsToChange,
        print "Random Pattern =",
        print getNumericNumForPattern(randomPatternToTest, patterns),
        print " ",
        print "Output After Error Induced =",
        print outputWithErroredPattern,
        print " ",
        print "Output After Missing Vals Inserted =",
        print outputWithMissingValsPattern,
        print "\n"
        '''

    '''
     Part-(c)
     Analysis on Spurious Patterns. These are patterns that a network recalls, but not in training set.
     Spurious patterns can be negation of the original pattern or linear combination of input vectors
     1. Negation of all Input Patterns
     2. Linear Combinations of Input patterns


    optimalTraining = [patterns[1], patterns[2], patterns[4], patterns[5], patterns[7], patterns[8]]
    weightMatrix = computeWeightMatrixUsingHebbRule(optimalTraining)

    for pattern in optimalTraining:
        flippedPattern = np.array(-pattern)
        outputWithSpuriousPattern = runOutputLayerTransferFunc(getOutputFromLayer(weightMatrix, flippedPattern))

        print "Original Pattern =",
        print getNumericNumForPattern(pattern, patterns)
        print " ",
        print "Is the Flipped pattern comes out same as was inserted",
        print checkIfPatternIsSame(flippedPattern, outputWithSpuriousPattern)

    optimalTraining = [patterns[1], patterns[2], patterns[4], patterns[5], patterns[7], patterns[8]]
    weightMatrix = computeWeightMatrixUsingHebbRule(optimalTraining)

    allCombinationsOfThreePatterns = itertools.combinations(range(0, 6), 3)
    for comb in allCombinationsOfThreePatterns:
        summedPattern = optimalTraining[comb[0]] + optimalTraining[comb[1]] + optimalTraining[comb[2]]
        #createVizForPatterns(summedPattern)

        outputWithSpuriousPattern = runOutputLayerTransferFunc(getOutputFromLayer(weightMatrix, summedPattern))
        #createVizForPatterns(outputWithSpuriousPattern)
        print comb,
        print " ",
        print "Is same as Original Pattern =",
        print checkIfPatternIsSame(summedPattern, outputWithSpuriousPattern)
    '''

if __name__ == "__main__":
    main()