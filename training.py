from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
LE = LabelEncoder()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle
from sklearn.pipeline import make_pipeline
from mlflow.tracking import MlflowClient
from sklearn.svm import SVC
from pickle import dump
from pickle import load
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from encode import Custom_encoder_data
from sklearn.metrics import accuracy_score
import warnings
import mlflow
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.metrics import f1_score,precision_score,recall_score
from fastapi import FastAPI, File, UploadFile
from io import BytesIO

import mysql.connector
### connect with mysql 
def dbconnect():
    mydb = mysql.connector.connect(host="localhost",
    user="root",
    password="admin",
    database="loop_template_plant",
    port = '3306'
    )
    return mydb
### fetch the data from database
def fetchdata(mydb,plant_name):
    mycursor = mydb.cursor()
    sql = "SELECT feature_name,is_dependent FROM Data WHERE plant_name = %s"
    adr = (plant_name, )
    mycursor.execute(sql, adr)
    myresult = mycursor.fetchall()
    return myresult

### fetch the list of feature list
def feature(data):
    feature_list =[]
    dependent_feature = []
    for i in data:
        print(i)
        if i[1] == 'TRUE':
            dependent_feature.append(i[0])
        else:
           feature_list.append(i[0])
    return feature_list,dependent_feature 

train_file = r"training_data_unique.csv"
test_file = r"test_data.csv"
feature_list = ['INPUT SIGL TYPE_Mod', 'Sig_Identifier_Group','SYSTEM_Mod']
dependent_feature = ['StrategyTemplateName_Final']
feture_name = {'io': "INPUT SIGL TYPE_Mod",'sigi':"Sig_Identifier_Group",'syst':"SYSTEM_Mod"}
###   loading the traning data and testing data
#mlflow.start_run()
trdf = pd.read_csv(train_file)
tstdf = pd.read_csv(test_file)
#trdf = trdf[trdf['StrategyTemplateName_Final'] != '$DCS_DI_ALM']
X_train = trdf[feature_list]


y_train = trdf[dependent_feature]
X_test = tstdf[feature_list]
y_test = tstdf[dependent_feature]
#clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))


def train_test_data(file_path):
    if file_format == 'xlsx':
        df = pd.read_excel(file_path)
        #tstdf = pd.read_excel(test_file)
    elif file_format == 'csv':
        df = pd.read_csv(train_file)
        #tstdf = pd.read_csv(test_file)
    else:
        return "Wrong file format"
    return df

encode = Custom_encoder_data(feature_list)
X_train_new = encode.fit(X_train)
print("Len of uniques")
print(len(encode.uniques))
dump(encode, open('encode1.pkl', 'wb'))
X_test_new = encode.transform(X_test)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold,KFold
from tpot import TPOTClassifier
cv = KFold(n_splits=10,shuffle=True, random_state=1)

classifier_config_sparse = {

    'sklearn.neighbors.KNeighborsClassifier': {
        'n_neighbors': range(1, 101),
        'weights': ["uniform", "distance"],
        'p': [1, 2]
    },

    'sklearn.ensemble.RandomForestClassifier': {
        'n_estimators': [100],
        'criterion': ["gini", "entropy"],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf':  range(1, 21),
        'bootstrap': [True, False]
    },

    # 'sklearn.feature_selection.SelectFwe': {
    #     'alpha': np.arange(0, 0.05, 0.001),
    #     'score_func': {
    #         'sklearn.feature_selection.f_classif': None
    #     }
    # },

    # 'sklearn.feature_selection.SelectPercentile': {
    #     'percentile': range(1, 100),
    #     'score_func': {
    #         'sklearn.feature_selection.f_classif': None
    #     }
    # },

    # 'sklearn.feature_selection.VarianceThreshold': {
    #     'threshold': np.arange(0.05, 1.01, 0.05)
    # },

    # 'sklearn.feature_selection.RFE': {
    #     'step': np.arange(0.05, 1.01, 0.05),
    #     'estimator': {
    #         'sklearn.ensemble.ExtraTreesClassifier': {
    #             'n_estimators': [100],
    #             'criterion': ['gini', 'entropy'],
    #             'max_features': np.arange(0.05, 1.01, 0.05)
    #         }
    #     }
    # },

    # 'sklearn.feature_selection.SelectFromModel': {
    #     'threshold': np.arange(0, 1.01, 0.05),
    #     'estimator': {
    #         'sklearn.ensemble.ExtraTreesClassifier': {
    #             'n_estimators': [100],
    #             'criterion': ['gini', 'entropy'],
    #             'max_features': np.arange(0.05, 1.01, 0.05)
    #         }
    #     }
    # },

    'sklearn.linear_model.LogisticRegression': {
        'penalty': ["l1", "l2"],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.],
        'dual': [True, False]
    },

    'sklearn.naive_bayes.BernoulliNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.naive_bayes.MultinomialNB': {
        'alpha': [1e-3, 1e-2, 1e-1, 1., 10., 100.],
        'fit_prior': [True, False]
    },

    'sklearn.svm.LinearSVC': {
        'penalty': ["l1", "l2"],
        'loss': ["hinge", "squared_hinge"],
        'dual': [True, False],
        'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        'C': [1e-4, 1e-3, 1e-2, 1e-1, 0.5, 1., 5., 10., 15., 20., 25.]
    },

    'xgboost.XGBClassifier': {
        'n_estimators': [100],
        'max_depth': range(1, 11),
        'learning_rate': [1e-3, 1e-2, 1e-1, 0.5, 1.],
        'subsample': np.arange(0.05, 1.01, 0.05),
        'min_child_weight': range(1, 21),
        'n_jobs': [1],
        'verbosity': [0]
    }
}

mlflow.set_experiment("/self_evolving")
mlflow.start_run()

model = TPOTClassifier(generations=6, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1,warm_start = True)
model.fit(X_train_new,y_train)
print("best model :",end=" ")
print(model.fitted_pipeline_)
y_prediction =model.fitted_pipeline_.predict(X_test_new)
print('Accuary:',model.fitted_pipeline_.score(X_test_new,y_test))
#print(dir(model.fitted_pipeline_))
print('f1_score:',f1_score(y_prediction,y_test,average='weighted'))
print('Precision Score:',precision_score(y_prediction,y_test,average='weighted'))
print('Recall:',recall_score(y_prediction,y_test,average='weighted'))
print(model.fitted_pipeline_._estimator_type)
print(model.fitted_pipeline_._final_estimator)
print(model.fitted_pipeline_._final_estimator.get_params())
mlflow.log_param("Model_details ", str(model.fitted_pipeline_._final_estimator).split('(')[0])
for i in model.fitted_pipeline_._final_estimator.get_params():
    mlflow.log_param(i,model.fitted_pipeline_._final_estimator.get_params()[i])
accuracy = model.fitted_pipeline_.score(X_test_new,y_test)
mlflow.log_metric('Accuary',accuracy)
F1_Score = f1_score(y_prediction,y_test,average='weighted')
mlflow.log_metric('f1_score',F1_Score)
#mlflow.log_metric('Precision Score:',precision_score(y_prediction,y_test,average='weighted'))
#mlflow.log_metric('Recall:',recall_score(y_prediction,y_test,average='weighted'))

dump(model.fitted_pipeline_, open('model.pkl', 'wb'))
mlflow.log_artifact('model.pkl')
mlflow.end_run()

