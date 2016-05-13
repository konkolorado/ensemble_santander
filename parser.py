"""
Uriel Mandujano
A parser for the Santander data set
"""

import numpy as np

def main():
    """
    storage = "/scratch/umanduj2/cs63/data/"
    training = storage + "santander_train.csv"
    testing = storage + "santander_test.csv"
    
    features = getFeatureLabels(training)
    userData = readFile(training)
    """
    ps = ParseSantander()
    features = ps.getFeatureLabels()
    usersToData = ps.readFile(ps.training)
    print "Number of training examples", len(usersToData)
    print "Number of features", len(features)
    
    counter = dict()
    for data in usersToData.values():
        data = data[0]
        for i, value in enumerate(data):
            if str(i) not in counter:
                counter[str(i)] =  set()
            counter[str(i)].add(value)
    print sum(1 for x in counter.values() if len(x) < 5)


class ParseSantander(object):
    def __init__(self, data_directory="/scratch/umanduj2/cs63/data/"):
        self.training = data_directory + "santander_train.csv"
        self.testing = data_directory + "santander_test.csv"

    def getFeatureLabels(self):
        """
        Opens a provided feature file and returns the labels for each feature
        dimension
        """
        inFile = open(self.training, 'r')
        features = inFile.readline().strip().split(',')
        inFile.close()
        return features[1:]

    def readFile(self, filename):
        """
        Opens a given file and stores the data into a dictionary where the 
        key is the user ID and the value is an array of their feature values
        """
        usersToData = dict()

        inFile = open(filename, 'r')
        features = inFile.readline()
        for line in inFile:
            line = line.strip().split(',')
            user, result, line = line[0], line[-1], line[1:len(line)-1]
            result = int(result)
            self.toInt(line)
            line = self.toArray(line)
            
            usersToData[user] = [line, result]

        inFile.close()
        return usersToData

    def readTest(self):
        """
        Reads the data in the test file available and puts it into the 
        proper format for evaluation
        """
        testData = dict()
        inFile = open(self.testing, 'r')
        features = inFile.readline()
        for line in inFile:
            line = line.strip().split(',')
            user, line = line[0], line[1:]
            self.toInt(line)
            line = self.toArray(line)
            
            testData[user] = line

        inFile.close()
        return testData

    def toInt(self, ls):
        """
        Given a list, sets each item in the list to be a float
        """
        for i in range(len(ls)):
            ls[i] = float(ls[i])

    def toArray(self, ls):
        """
        Returns the given list as a numpy array
        """
        return np.asarray(ls)

if __name__ == '__main__':
    main()
