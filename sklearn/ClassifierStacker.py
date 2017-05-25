#
# New BSD License
# 
# Copyright (c) 2017 Fausto Guzzo da Costa
# All rights reserved.
#

# github.com/faustogc/pyml/sklearn/ClassifierStacker.py
# version 1.0

#
# Scikit-learn handler for stacking classifiers
#

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

class ClassifierStacker():
    parms = dict()
    lr = None
    
    
    def __init__(self, clfs, clf_names,
                 first_layer_train_size=0.8,
                 with_features=False,
                 second_layer_classifier=LogisticRegression()):
        
        self.parms['clfs'] = clfs
        self.parms['clf_names'] = clf_names
        self.parms['first_layer_train_size'] = first_layer_train_size
        self.parms['with_features'] = with_features
        self.parms['second_layer_classifier'] = second_layer_classifier
 

    def fit(self, X, y):
        X_train1, X_train2, y_train1, y_train2 = train_test_split(X, y,
                                                                  train_size=self.parms['first_layer_train_size'])
        
        self._fit_first_layer(X_train1, y_train1)
        
        self._fit_second_layer(X_train2, y_train2)
     
    
    def _fit_first_layer(self, X, y):
        for clf in self.parms['clfs']:
            clf.fit(X, y)
    
    
    def _fit_second_layer(self, X, y):
        features = self._first_layer_predict(X, 'predict_proba')
        
        self.parms['second_layer_classifier'].fit(features, y)

        
    def predict_proba(self, X):
        features = self._first_layer_predict(X, 'predict_proba')
        
        return(self.parms['second_layer_classifier'].predict_proba(features))
    
    
    def predict(self, X):
        features = self._first_layer_predict(X, 'predict')
        
        return(self.parms['second_layer_classifier'].predict(features))
    
        
    def _first_layer_predict(self, X, func):
        preds = []
        
        for clf in self.parms['clfs']:
            predfunc = getattr(clf, func)
            clf_preds = predfunc(X)
            
            if clf_preds.ndim > 1:
                clf_preds = clf_preds[:,1]
            
            preds.append( clf_preds )
        
        features = pd.DataFrame(preds).T
        features.columns = self.parms['clf_names']
        
        if self.parms['with_features'] == True:
            features = pd.concat([X.reset_index(drop=True),
                                  features.reset_index(drop=True)], axis=1)
        
        return(features)
    
    
    def get_params(self, deep=False):
        return(self.parms)
    
    
    def corr(self, X):
        preds = self._first_layer_predict(X, 'predict_proba')
        
        return(sns.heatmap(preds.corr()))

