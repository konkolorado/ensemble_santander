"""
Uriel Mandujano
An ensemble classifier which attempts to predict customer satisfaction on 
the Santander dataset on Kaggle.

Uses the following models (available in Scikit-Learn) to create a prediction:
    K Nearest Neighbors
    K Means
    Support Vector Machines
    Neural Networks
"""

import pca

from sklearn import svm, neighbors, mixture, cross_validation
from sklearn.cluster import KMeans
#from sklearn.neural_network import MLPClassifier

import numpy as np
import random

increase = True
decrease = False

class Ensemble(object):
    def __init__(self, preprocessor="", k=1, m=2):
        # Initialize models
        self.models = {}
        self.models["SVM"] = svm.SVC()
        self.models["KNN"] = neighbors.KNeighborsRegressor(k, weights="distance")
        self.models["KMEANS"] = KMeans(n_clusters=m, random_state=99)
        #self.models["NN"] = MLPClassifier(algorithm='adam', \
        #                    hidden_layer_sizes=(100, 25, 3), random_state=1)
        self.models["GMM"] = mixture.GMM()
        
        # Set preprocessor data
        self.prepro = preprocessor
        self.pca = pca.PrincipalComponentAnalysis(self.prepro)
        self.feature_labels = self.pca.feature_labels
        self.users_data = self.pca.users_data
        self.testData = self.pca.test_data
        self.n_components = self.pca.n_components_pca
        self.scores = self.pca.scores

        # Prepare and extra relevant training/test data
        self.data = []
        self.labels = []#, np.array([])
        self.getPCAFeatures()
        self.cleanTestData()
        print "Preprocessing step is", self.prepro
        
        # Alter amount of positive / negtive training examples
        if increase:
            self.increaseNegativeExamples(self.data, self.labels)
        if decrease: 
            self.decreasePositiveExamples(self.data, self.labels)

        # Create folds for cross validation
        self.kf = cross_validation.KFold(len(self.data), n_folds=5)

    def train(self):
        """
        Fits the models and performs cross validation to predict
        """
        self.runSVM()
        self.runKNN()
        self.runKMeans()
        self.runGMM()
   
    def getPCAFeatures(self):
        """
        Given PCA scores and components, extract those features in the data 
        corresponding to the highest scoring features in PCA. This is our new 
        dataset.
        """
        # Get the top n most salient features' indices according to PCA
        self.indices = np.array(self.scores).argsort()[::-1][:self.n_components]
        
        data, labels = [], np.array([])
        for user in self.users_data:
            data.append(self.users_data[user][0])
            self.labels = np.append(self.labels, self.users_data[user][1])
        
        # Create a new data set containing only the most salient features
        for d in range(len(data)):
            new_data = []
            for i in self.indices:
                new_data.append(data[d][i])
                
            data[d] = new_data
        
        # Process the data and update class variable
        data = self.processData(data)
        self.data = data

    def cleanTestData(self):
        """
        Processes the test data and extracts the relevant features from the 
        set.
        """
        test_data, self.test_users = [], []
        for user in self.testData:
            self.test_users.append(user)
            test_data.append(self.testData[user])

        # Get the relevant features
        for d in range(len(test_data)):
            new_test_data = []
            for i in self.indices:
                new_test_data.append(test_data[d][i])
            
            test_data[d] = new_test_data

        # Process test_data and update class variable 
        test_data = self.processData(test_data)
        self.testData = test_data

    def processData(self, X):
        """
        Processes the current data so that it is suitable for comparison 
        with the saved feature scores
        """
        if self.prepro == "scale":
            X = self.pca.scale(X)
        if self.prepro == "normalize":
            X = self.pca.normalize(X)
        if self.prepro == "sparse":
            X = self.pca.sparse(X)
        return X
  
    def getAccuracy(self, true_labels, predictions):
        """
        Given a list of correct labels and predictions, returns the accuracy 
        as percent correct.
        """
        correct = 0.0
        fp, fn, tp, tn = 0, 0, 0, 0
        for i,j in zip(true_labels, predictions):
            if i == j:
                correct += 1
            if i == 0 and j == 0:
                tp += 1
            if i == 0 and j == 1:
                fp += 1
            if i == 1 and j == 0:
                fn += 1
            if i == 1 and j == 1:
                tn += 1

        total = float(tp+fp+fn+tn)
        print "\n/" + "-" * 45 + "/"
        print "false positives:", fp, "(", fp/total, ")"
        print "false negatives:", fn, "(", fn/total, ")"
        print "true positives:", tp, "(", tp/total , ")"
        print "true negatives:", tn, "(", tn/total , ")"
        print "/" + "-" * 45 + "/\n"

        return correct / len(true_labels)

    def getCurrFoldTrainData(self, train_indices):
        """
        Given a list of indices, return the associated training data
        """
        return [self.data[x] for x in train_indices], \
               [self.labels[x] for x in train_indices]

    def getCurrFoldTestData(self, test_indices):
        """
        Givena list of indices, return the associated test data
        """
        return [self.data[x] for x in test_indices], \
               [self.labels[x] for x in test_indices]

    def runSVM(self):
        """
        Runs the SVM on 5 different splits of cross validation data
        """
        for train, test in self.kf:
            svm = self.models["SVM"]

            train_set, train_labels = self.getCurrFoldTrainData(train)
            test_set, test_labels = self.getCurrFoldTestData(test)
            svm.fit(train_set, train_labels)

            preds = svm.predict(test_set)
            acc = self.getAccuracy(test_labels, preds)
            print "(SVM) Percent correct is", acc
    
    def runKNN(self):
        """
        Runs KNN on 5 different split of cross validation data
        Best KNN is 1
        """
        for train, test in self.kf:
            knn = self.models["KNN"]
            
            train_set, train_labels = self.getCurrFoldTrainData(train)
            test_set, test_labels = self.getCurrFoldTestData(test)
            knn.fit(train_set, train_labels)

            # score = knn.score(test_set, test_labels)
            # NOTE Score always seems to be equal to zero. This implies that 
            # the model always predicts the same value of y, disregarding 
            # the input features
            preds = knn.predict(test_set)
            acc = self.getAccuracy(test_labels, preds)
            print "(KNN) Percent correct is", acc

    def runKMeans(self):
        """
        Runs KMeans on 5 different split of cross validation data
        """
        for train, test in self.kf:
            kmeans = self.models["KMEANS"]

            train_set, train_labels = self.getCurrFoldTrainData(train)
            test_set, test_labels = self.getCurrFoldTestData(test)

            kmeans.fit(train_set, train_labels)
            preds = kmeans.predict(test_set)
            acc = self.getAccuracy(test_labels, preds)
            print "(KMeans) Percent correct is", acc 

    def runGMM(self):
        """
        Runs GMM model on 5 different splits of cross validation data
        """
        for train, test in self.kf:
            gmm = self.models["GMM"]

            train_set, train_labels = self.getCurrFoldTrainData(train)
            test_set, test_labels = self.getCurrFoldTestData(test)

            gmm.fit(train_set, train_labels)
            preds = gmm.predict(test_set)
            acc = self.getAccuracy(test_labels, preds)
            print "(GMM) Percent correct is", acc

    def runEnsemble(self):
        """
        Predicts the target label for a feature vector by combining and 
        weighting the predictions of the individual classifiers
        """
        for train, test in self.kf:
            # Extract models
            knn = self.models["KNN"]
            kmeans = self.models["KMEANS"]
            svm = self.models["SVM"]
            gmm = self.models["GMM"]
            
            # Set up training and test data
            train_set, train_labels = self.getCurrFoldTrainData(train)
            test_set, test_labels = self.getCurrFoldTestData(test)
            
            if increase:
                train_set, train_labels=self.subsetData(train_set, train_labels)
            
            # Fit the models
            knn.fit(train_set, train_labels)
            kmeans.fit(train_set, train_labels)
            svm.fit(train_set, train_labels)
            gmm.fit(train_set, train_labels)

            # Generate predictions by weighting each model using accuracies 
            # created from earlier runs
            knn_pred = knn.predict(test_set)
            kmeans_pred = kmeans.predict(test_set)
            svm_pred = svm.predict(test_set)
            gmm_pred = gmm.predict(test_set)
            
            preds = self.weightPredictions(knn_pred, kmeans_pred, \
                                           svm_pred, gmm_pred)
            acc = self.getAccuracy(test_labels, preds)
            print "(ENSEMBLE) Percent correct is", acc

    def weightPredictions(self,knn, kmeans, svm, gmm):
        """
        Given a list of predictions that each learning method predicted, 
        create a final list of predictions.
        """
        """
        WEIGHTS:
            SVM: .28
            KNN: .27
            KMEANS: .17
            GMM: .28
        NOTES:
            >= 2: Avg Acc is .9484
            >= 3: Avg Acc is .96043
            >= 1: Avg Acc is .5769
            >= Weighted .75 Avg Acc is .96043
            >= Weighted .5 Avg Acc is .96043 
            >= Weighted .4 Avg Acc is .9484
        With Augmented Negative Data:
            >= Weighted .4 Avg Acc is .9607 (mode is about .98)!
            >= Weighted .5 Avg Acc is .90 (mode is .99)!
        """
        preds = []
        for i in range(len(knn)):
            if .27*knn[i] + .17*kmeans[i] + .28*svm[i] + .28*gmm[i] >= .5:
                preds.append(1)
            else:
                preds.append(0)
        return preds

    def subsetData(self, trainData, trainLabels):
        """
        This method should only be called when the data has been artificially 
        augmented. Shuffles and cuts the training data to be 7000 examples. 
        Returns datasets
        """
        training = zip(trainData, trainLabels)
        random.shuffle(training)
        train_set, train_labels = zip(*training)
        train_set, train_labels = train_set[:7000], train_labels[:7000]
        return train_set, train_labels

    def decreasePositiveExamples(self, trainData, trainLabels):
        """
        Santander data is heavily skewed because it is mostly examples of 
        positive examples. This function ranomdly removes positive training 
        examples until the number of positive examples is equal to the number 
        of negative examples
        """
        print "<Decreasing positive training examples>"

        positive_indices = []
        negative_count = 0
        for i, label in enumerate(trainLabels):
            # Indicates a positive label
            if label == 0:
                positive_indices.append(i)
            if label == 1:
                negative_count += 1
       
        positive_keepers = set(random.sample(positive_indices, negative_count))
        new_train_data, new_train_labels = [], []
        for i in positive_keepers:
            new_train_data.append(trainData[i])
            new_train_labels.append(trainLabels[i])
        
        for i in range(len(trainData)):
            if trainLabels[i] == 1:
                new_train_data.append(trainData[i])
                new_train_labels.append(trainLabels[i])
        
        self.data = new_train_data
        self.labels = new_train_labels

    def increaseNegativeExamples(self, trainData, trainLabels):
        """
        Santander data is heavily skewed because it is mostly examples of 
        positive examples. This function duplicates the negative training 
        examples so that there are more negative examples to learn from
        """
        print "<Increasing negative examples>"

        negative_indices = []
        positive_count = 0
        for i, label in enumerate(trainLabels):
            # Indicates negative labels
            if label == 1:
                negative_indices.append(i)
            if label == 0:
                positive_count += 1

        negative_count = len(negative_indices)
        curr = 0
        while negative_count < positive_count:
            noised_data = self.addNoise(trainData[negative_indices[curr]])
            trainData = np.append(trainData, [noised_data], axis=0)
            trainLabels = np.append(trainLabels, \
                                           trainLabels[negative_indices[curr]])
            negative_count += 1
            curr += 1
            if curr == len(negative_indices):
                curr = 0
            
        self.data = trainData
        self.labels = trainLabels

    def addNoise(self, data):
        """
        Adds uniform noise to a data point. Meaned around 0 with a standard 
        deviation of 1. Could create more reliable data
        """
        noise = np.random.normal(0,1,len(data))
        for i in range(len(noise)):
            data[i] += noise[i]
        return data

    def savePredictions(self, X, users):
        """
        Given an array of predictions and the corresponding user we are 
        predicting, write the predictions to a file in the proper format 
        for submission to the Kaggle contest.
        """
        outfile = open("sant_submission.csv", "w")
        outfile.write("ID,TARGET\n")
        for user, pred in zip(users, X):
            outfile.write(str(user) + "," + str(int(pred)) + "\n")
        outfile.close()

def main():
    e = Ensemble("scale",k=1, m=2)
    #e.train()
    #e.test()
    e.runEnsemble()

if __name__ == '__main__':
    main()
