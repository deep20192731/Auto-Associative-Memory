import os
import sys
import numpy as np

INPUT_FILE_NAME = "TenDigitPatterns.txt"
INPUT_PATTERN = '#'
PATTERN_VECTOR_LEN = 35

THRESHOLD = 0 # Threshold used at output layer

def readInputFile(fileName):
    filHandle = open(fileName)
    return map(lambda pattern:pattern[:PATTERN_VECTOR_LEN], filHandle.readlines())

def areVectorsOrthogonal(vec1, vec2): # Length of both vectors is assumed to be same.
    sum = 0
    for i in range(0, len(vec1)):
        sum = sum + vec1[i]*vec2[i] # dot-product
    if(sum == 0): return True
    else: False

def getNumericNumForPattern(pattern, patterns): # assuming PATTERNS Vector is already filled
    match = -1
    for patternNum in range(0, len(patterns)):
        if(len(pattern) != len(patterns[patternNum])):
            return match
        else:
            uncommon = [1 for x,y in zip(pattern, patterns[patternNum]) if x != y]
            if(len(uncommon) == 0): return patternNum
        
    return match
    

def convertPatternToVector(pattern):
    return np.array(map(lambda x: 1 if x == INPUT_PATTERN else -1, pattern)) # using bipolar vectors

def convertPatternsToInputVectors(patterns):
    inputMatrix = np.zeros(shape = (len(patterns), PATTERN_VECTOR_LEN))
    for i in range(0, len(patterns)):
        inputMatrix[i] = convertPatternToVector(patterns[i])
    return inputMatrix

'''
    f(x) = {
            +1 if x > THRESHOLD
            0 if -THRESHOLD <= x <= THRESHOLD
            -1 if x < -THRESHOLD
            }
'''
def runOutputLayerTransferFunc(outputFromNet):
    out = []
    for i in range(0, len(outputFromNet)):
        if(outputFromNet[i] > THRESHOLD): out.append(1)
        elif(outputFromNet[i] < -THRESHOLD): out.append(-1)
        else: out.append(0)
    return out


def constructOrthogonalMatrix(patterns):
    mat = np.zeros(shape = (len(patterns), len(patterns)))
    counter = 0
    for outerPattern in patterns:
        l = []
        for innerPattern in patterns:
            l.append(areVectorsOrthogonal(innerPattern, outerPattern))
        mat[counter] = l[counter]
        counter = counter + 1
    return mat
    
def getOutputFromLayer(weightMatrix, inputPattern): # no pre-checks on size. Assumed all of right dimensions
    return weightMatrix.dot(inputPattern)

def computeWeightMatrixUsingHebbRule(patterns):
    weightMatrix = np.zeros(shape = (PATTERN_VECTOR_LEN, PATTERN_VECTOR_LEN))
    for pattern in patterns:
        weightMatrix += map(lambda x: pattern if x == 1 else -pattern, pattern)

    return weightMatrix        

def main():
    patterns = readInputFile(INPUT_FILE_NAME) # Read the input patterns file
    patterns =  convertPatternsToInputVectors(patterns) # convert '#' pattern to input binary vector
    #orthogonalMatrix = constructOrthogonalMatrix(patterns)
    # print orthogonalMatrix
    
    pattern = patterns[3]
    weightMatrix = computeWeightMatrixUsingHebbRule([patterns[0], patterns[1], patterns[2], patterns[3]])
    result = getNumericNumForPattern(runOutputLayerTransferFunc(getOutputFromLayer(weightMatrix, pattern)), patterns)
    print result
    
    ''' Test Code uptil now
    patterns = np.zeros(shape = (2, 4))
    patterns[0] = np.array([1,1,-1,-1])
    patterns[1] = np.array([-1,1,1,-1])
    #print inputMat
    print computeWeightMatrixUsingHebbRule(patterns)
    '''
    
    #result = runOutputLayerTransferFunc(getOutputFromLayer(weightMatrix, patterns[9]))
    #print result
    #print getNumericNumForPattern(result, patterns)
    
if __name__ == "__main__":
    main()
