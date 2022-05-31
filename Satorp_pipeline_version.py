import os
import sklearn
#from tpot import TPOTClassfier
from tpot import TPOTClassifier
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV,RepeatedStratifiedKFold
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support


# Define features used for training
stn = "StrategyTemplateName_Final"
io = "INPUT SIGL TYPE_Mod"
sigi = "Sig_Identifier_Group"
syst = "SYSTEM_Mod"
subsyst = "SubSystem"
InstrumentType = "InstrumentType"
def list_unique_features(df,f_list):
    return list(set(tuple(sorted(sub)) for sub in [[j.strip() for j in i if j==j] for i in df[f_list].values.tolist()]))

# Create List of unique features from test data
def combine_features(df,f_list):
    return [[j.strip() for j in i if j==j] for i in df[f_list].values.tolist()]

# Assign availability score to test data based on availability of features
def feature_availability_score(trdf,tstdf,feature_list):
    global unique_features_training
    unique_features_training = list_unique_features(trdf,feature_list)
    tstdf['unique_features'] = combine_features(tstdf,feature_list)
    for i,row in tstdf.iterrows():
        score = []
        [score.append(len(set(row['unique_features']).intersection(feature)) / max(len(set(row['unique_features'])),len(set(feature)))) for feature in unique_features_training]
        test_features = [ item for elem in [i.split(',') for i in row['unique_features']] for item in elem]
        train_features = unique_features_training[score.index(max(score))]
        train_features = [ item for elem in [i.split(',') for i in train_features] for item in elem]
        tstdf.at[i,'Availability_score'] = len(set(test_features).intersection(train_features)) / max(len(set(test_features)),len(set(train_features)))
    del tstdf['unique_features']

# Import training and test data
train_file = r"training_data_unique.csv"
test_file = r"test_data.csv"
feature_list = ['INPUT SIGL TYPE_Mod', 'Sig_Identifier_Group','InstrumentType','SYSTEM_Mod', 'SubSystem']
dependent_feature = ['StrategyTemplateName_Final']
trdf = pd.read_csv(train_file)
tstdf = pd.read_csv(test_file)
trdf = trdf[feature_list+dependent_feature]
test_df = tstdf[feature_list+dependent_feature]

# define evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search
model = TPOTClassifier(generations=5, population_size=50,  scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1)
cols_to_encode = feature_list
trdf[pd.isnull(trdf)]  = ''
uniques = {k for c in cols_to_encode for i in list(trdf[c].unique()) for k in str(i).lower().split(",")}
uniques = sorted(uniques)
uniques.append(dependent_feature[0])
def encode_data(df):
    def f():
        return {i: 0 for i in uniques}
    rows = []
    for i, r in df.iterrows():
        nr = f()
        nr[stn] = r[stn]
        d = Counter([k for c in cols_to_encode for k in str(r[c]).lower().split(',')])
        nr.update(d)
        d = [nr[k] for k in uniques]
        rows.append(d)
    return pd.DataFrame(rows, columns=uniques)   
train_df = encode_data(trdf)
test_df = encode_data(test_df)
x_train, y_train = train_df[uniques[:-1]], train_df[uniques[-1]]
x_test, y_test = test_df[uniques[:-1]], test_df[uniques[-1]]

# perform the search
model.fit(x_train, y_train)
# export the best model
model.export('tpot_sonar_best_model.py')