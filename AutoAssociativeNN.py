import os
import sys
import numpy as np

INPUT_FILE_NAME = "TenDigitPatterns.txt"
INPUT_PATTERN = '#'
PATTERN_VECTOR_LEN = 35

def readInputFile(fileName):
    filHandle = open(fileName)
    return map(lambda pattern:pattern[:PATTERN_VECTOR_LEN], filHandle.readlines())

def convertPatternToVector(pattern):
    return np.array(map(lambda x: 1 if x == INPUT_PATTERN else -1, pattern)) # using bipolar vectors

def convertPatternsToInputVectors(patterns):
    inputMatrix = np.zeros(shape = (len(patterns), PATTERN_VECTOR_LEN))
    for i in range(0, len(patterns)):
        inputMatrix[i] = convertPatternToVector(patterns[i])
    return inputMatrix

def computeWeightMatrixUsingHebbRule(patterns):
    weightMatrix = np.zeros(shape = (PATTERN_VECTOR_LEN, PATTERN_VECTOR_LEN))
    for pattern in patterns:
        weightMatrix += map(lambda x: pattern if x == 1 else -pattern, pattern)

    return weightMatrix        

def main():
    patterns = readInputFile(INPUT_FILE_NAME) # Read the input patterns file
    inputMat =  convertPatternsToInputVectors(patterns) # convert # pattern to input binary vector
    computeWeightMatrixUsingHebbRule(inputMat)
    
    ''' Test Code
    patterns = np.zeros(shape = (2, 4))
    patterns[0] = np.array([1,1,-1,-1])
    patterns[1] = np.array([-1,1,1,-1])
    #print inputMat
    print computeWeightMatrixUsingHebbRule(patterns)
    '''

if __name__ == "__main__":
    main()
