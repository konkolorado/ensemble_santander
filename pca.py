"""
Uriel Mandujano
A program for performing Principal Component Analysis on the 
training data provided by Santander competition on Kaggle
"""

from scipy import linalg
from sklearn import preprocessing
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cross_validation import cross_val_score
import numpy as np

import cPickle as pk
import parser
import sys
import os

class PrincipalComponentAnalysis(object):
    def __init__(self, preprocessing=""):
        ps = parser.ParseSantander()
        self.feature_labels = ps.getFeatureLabels()
        self.users_data = ps.readFile(ps.training)
        self.test_data = ps.readTest()

        self.preprocessor = preprocessing
        # n_components indicates the top n features we want to identify 
        self.n_components = np.arange(0, len(self.feature_labels), 5)
        
        # better scores are most positive
        self.scores = []
        self.n_components_pca = 0
   
        self.run()
    
    def run(self):
        """
        Performs the full principal component analysis task
        """
        if self.loadData():
            print "<Loading data>"
            return

        data = [x[0] for x in self.users_data.values()]
        data = self.preprocess(data)

        pca_scores = self.computeScores(data)  
        n_components_pca = self.n_components[np.argmax(pca_scores)]

        print "pca_scores", pca_scores
        print "n_componenets_pca", n_components_pca
        
        self.writeData(pca_scores, n_components_pca)

    def computeScores(self, X):
        """
        Computes the scores for a given X feature vector considering various
        numbers of features
        """
        pca = PCA()
        pca_scores = []

        for n in self.n_components:
            print "Computing score for", n, "components"
            sys.stdout.flush()
            
            pca.n_components = n
            pca_scores.append(np.mean(cross_val_score(pca, X)))

        return pca_scores

    def writeData(self, scores, n_components):
        """ 
        Saves the scores for each feature dimension as an array as well as the 
        number of ideal components determined by PCA. Saved as a pickle file.
        """
        save_location = "data/"
        scores_file = save_location + self.preprocessor + "_scores.pk"
        components_file = save_location + self.preprocessor + "_components.pk"

        if not os.path.isdir(save_location):
            os.makedirs(save_location)

        with open(scores_file, "wb") as f:
            pk.dump(scores, f)

        f.close()

        with open(components_file, "wb") as f:
            pk.dump(n_components, f)

        f.close()

    def loadData(self):
        """ 
        Loads pre-existing PCA data. Saves unnecessary computation time
        """
        scores_file = "data/" + self.preprocessor + "_scores.pk"
        components_file = "data/" + self.preprocessor + "_components.pk"

        if (not os.path.exists(scores_file)) or \
                (not os.path.exists(components_file)):
            print "Attempted to load non-existant data. Will run PCA"
            return False

        self.scores = pk.load(open(scores_file, "rb"))
        self.n_components_pca = pk.load(open(components_file, "rb"))
        return True

    def preprocess(self, X):
        """
        Performs preprocessing on the data. If none is specified, no 
        preprocessing occurs. 
        """
        print "Preprocessing using", self.preprocessor
        if self.preprocessor == "scale":
            X = self.scale(X)
        if self.preprocessor == "normalize":
            X = self.normalize(X)
        if self.preprocessor == "sparse":
            X = self.sparse(X)
        return X

    def scale(self, X):
        """
        This function standardizes the values of a feature. Experimental use 
        only. Final scores will determine whether or not this is a useful 
        preprocessing step
        """
        return preprocessing.scale(X)

    def normalize(self, X):
        """
        This function normalizes the values for a feaure. Experimental use 
        only. Final scores will determine whether or not this is a useful 
        preprocessing step
        """
        return preprocessing.normalize(X, norm='l2')

    def sparse(self, X):
        """
        This function performs a preprocessing on features which retains 
        the sparsity of features. A lot of the data is 0 which probably means 
        it's missing. Experimental use only.
        """
        return preprocessing.maxabs_scale(X)

def main():
    pca = PrincipalComponentAnalysis("scale")
    print pca.n_components_pca
if __name__ == '__main__':
    main()
