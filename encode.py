from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from collections import Counter
import pandas as pd
import numpy as np

class Custom_encoder_data(BaseEstimator,TransformerMixin):
    def __init__(self,feature_list):

        self.uniques = []
        self.feature_list = feature_list
        #self.feature_name = feature_name
        #print("INit")

    def f(self):
        return {i: 0 for i in self.uniques}

    def fit(self ,X,y=None):
        X_copy = X.copy()
        X_copy[pd.isnull(X_copy)]  = np.nan
        self.uniques = {k for c in self.feature_list for i in list(X_copy[c].unique()) for k in str(i).lower().split(",")}
        self.uniques = sorted(self.uniques)
        rows = []
        for i, r in X_copy.iterrows():
            nr = self.f()
            #nr[self.feature_name['stn']] = r[self.feature_name['stn']]
            d = Counter([k for c in self.feature_list for k in str(r[c]).lower().split(',')])
            nr.update(d)
            d = [nr[k] for k in self.uniques]
            rows.append(d)
            
        return pd.DataFrame(rows, columns=self.uniques)


    def transform (self,X,y=None):
        rows = []
        for i, r in X.iterrows():
            nr = self.f()
            #nr[self.feature_name['stn']] = r[self.feature_name['stn']]
            d = Counter([k for c in self.feature_list for k in str(r[c]).lower().split(',')])
            nr.update(d)
            d = [nr[k] for k in self.uniques]
            rows.append(d)
        return pd.DataFrame(rows, columns=self.uniques)



        