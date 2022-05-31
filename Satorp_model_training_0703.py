
# Import Libraries
import os
import pandas as pd
from collections import Counter
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

# Define features used for training
stn = "StrategyTemplateName_Final"
io = "INPUT SIGL TYPE_Mod"
sigi = "Sig_Identifier_Group"
syst = "SYSTEM_Mod"
subsyst = "SubSystem"
InstrumentType = "InstrumentType"

# Change home directory path
#path = ''
#os.chdir(path)

# Import training and test data
train_file = r"training_data_unique.csv"
test_file = r"test_data.csv"
feature_list = ['INPUT SIGL TYPE_Mod', 'Sig_Identifier_Group','InstrumentType','SYSTEM_Mod', 'SubSystem']
dependent_feature = ['StrategyTemplateName_Final']
trdf = pd.read_csv(train_file)
tstdf = pd.read_csv(test_file)
trdf = trdf[feature_list+dependent_feature]
test_df = tstdf[feature_list+dependent_feature]
model_accuracy_df = pd.DataFrame(columns = ['Model_Name', 'CM_Accuracy','Cross_Val_Accuracy'])

# Create List of unique features from training data
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

feature_availability_score(trdf,tstdf,feature_list)
    
# Find Ambigious data
# ambigious_df = trdf.groupby([syst,subsyst,io,sigi,InstrumentType], as_index=False).agg(set)
#trdf = trdf.loc[~trdf['StrategyTemplateName_Final'].isin(['$DCS_PUMP_LR_CU','$ESD_DO_CU','$ESD_DI'])]

# Encode Categorical features
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
print(train_df.shape[1])
test_df = encode_data(test_df)
pickle.dump(train_df, open("encoded_training_data_satorp.pkl", "wb"))

x_train, y_train = train_df[uniques[:-1]], train_df[uniques[-1]]
x_test, y_test = test_df[uniques[:-1]], test_df[uniques[-1]]

# Define function to train and test data on each model
def train_model(model,model_name):
    '''
    Take a model as input.
    Train it on the training set.
    Predict test set results.
    Evaluate model using Confusion Matrix
    Save the model to .pkl file
    '''
    global model_accuracy_df
    model.fit(x_train, y_train)

    # Predicting the Test set results
    global y_pred, y_pred_proba
    y_pred = model.predict(x_test)
    pred_proba = model.predict_proba(x_test)
    [l.sort() for l in pred_proba]
    pred_diff = [l[-1] - l[-2] for l in pred_proba]
    y_pred_proba = pd.DataFrame(data=model.predict_proba(x_test), columns=model.classes_)
    y_pred_proba['prob_diff'] = pred_diff
    
    # Evaluating Model Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    score = accuracy_score(y_test,y_pred)
    print('Confusion Matrix : \n',cm,score*100)
    
    # Cross Validation Classification Accuracy
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    Cross_Val_Accuracy = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    print("Accuracy: %.3f (%.3f)" % (Cross_Val_Accuracy.mean(), Cross_Val_Accuracy.std()))

    model_accuracy_df = model_accuracy_df.append({'Model_Name' : model_name,
                                                  'CM_Accuracy' : score*100,
                                                  'Cross_Val_Accuracy':Cross_Val_Accuracy.mean()*100}, ignore_index = True)
    pr = pd.DataFrame({'class' : model_randomforest.classes_,
                       'Precision' : precision_recall_fscore_support(y_test, y_pred, average=None, labels=model_randomforest.classes_)[0],
                       'recall' : precision_recall_fscore_support(y_test, y_pred, average=None, labels=model_randomforest.classes_)[1],
                       'f_score':precision_recall_fscore_support(y_test, y_pred, average=None, labels=model_randomforest.classes_)[2]})

# Fitting SVM to the Training set
'''
from sklearn.svm import SVC
model_svc_sigmoid = SVC(C= 10, gamma= 0.01, kernel= 'rbf')
train_model(model_svc_sigmoid,"SVC")
'''
'''
# Hyper Parameter Tunning
model_svc_sigmoid.get_params()
parameters = {'C': [0.1,1, 10, 100], 
              'gamma': [1,0.1,0.01,0.001],
              'kernel': ['rbf', 'poly', 'sigmoid']
              }

clf = GridSearchCV(SVC(), parameters,n_jobs=-1)
clf.fit(x_train, y_train)
print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
print('Best Parameters : ',clf.best_params_)

Best Parameters :  {'C': 10, 'gamma': 0.01, 'kernel': 'rbf'}
'''

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
model_randomforest = RandomForestClassifier()
train_model(model_randomforest,"Random_Forest")
'''
# Hyper Parameter Tunning
model_randomforest.get_params()

parameters = { 'n_estimators': [10,20,50, 100, 200],
              'max_features': ['auto', 'sqrt', 'log2'],
              'max_depth' : [1,2,3,4,5,6,7],
              'criterion' :['gini', 'entropy'] }
clf = GridSearchCV(RandomForestClassifier(), parameters,n_jobs=-1)
clf.fit(x_train, y_train)
print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
print('Best Parameters : ',clf.best_params_)
    
Best Parameters :  {'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'n_estimators': 100}
'''

# Fitting Naive Bayes to the Training set
'''
from sklearn.naive_bayes import GaussianNB
model_GaussianNB = GaussianNB()
train_model(model_GaussianNB,"GaussianNB")
'''
'''
# Hyper Parameter Tunning
model_GaussianNB.get_params()

Best Parameters :  default
'''

# Fitting K-NN to the Training set
'''
from sklearn.neighbors import KNeighborsClassifier
model_KNN = KNeighborsClassifier(metric= 'minkowski', n_neighbors= 1, weights= 'uniform')
train_model(model_KNN,"KNN")
'''
'''
# Hyper Parameter Tunning
model_KNN.get_params()
parameters = {"n_neighbors":[1,3,5,10],
              "weights":['uniform','distance'],
              "metric":['minkowski','euclidean','manhattan']
              }

clf = GridSearchCV(KNeighborsClassifier(), parameters,n_jobs=-1)
clf.fit(x_train, y_train)
print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
print('Best Parameters : ',clf.best_params_)

Best Parameters :  {'metric': 'minkowski', 'n_neighbors': 1, 'weights': 'uniform'}
'''

# Fitting entropy to the Training set
'''
from sklearn.tree import DecisionTreeClassifier
model_dt = DecisionTreeClassifier(criterion= 'entropy', max_depth= 10)
train_model(model_dt,"Decision_Tree")
'''
'''
# Hyper Parameter Tunning
model_dt.get_params()
parameters = {"criterion":['entropy','gini'],
              "max_depth":[5,10,100]
              }

clf = GridSearchCV(DecisionTreeClassifier(), parameters,n_jobs=-1)
clf.fit(x_train, y_train)
print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
print('Best Parameters : ',clf.best_params_)

Best Parameters :  {'criterion': 'entropy', 'max_depth': 10}
'''

# Fitting AdaBoost to the Training set
'''
from sklearn.ensemble import AdaBoostClassifier
model_adaboost = AdaBoostClassifier(n_estimators=50,learning_rate=0.01)
train_model(model_adaboost,"AdaBoost")
'''
'''
# Hyper Parameter Tunning
model_adaboost.get_params()
parameters = {"n_estimators": [50],
              'learning_rate':[.001,0.01,.1,1]
              }

clf = GridSearchCV(AdaBoostClassifier(), parameters,n_jobs=-1)
clf.fit(x_train, y_train)
print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
print('Best Parameters : ',clf.best_params_)

Best Parameters :  {'learning_rate': 0.01, 'n_estimators': 50}
'''

# Creating a model using Multinomial Naive-Bayes
'''
from sklearn.naive_bayes import MultinomialNB
model_MultinomialNB = MultinomialNB(alpha= 1)
train_model(model_MultinomialNB,"MultinomialNB")
'''
'''
# Hyper Parameter Tunning
model_MultinomialNB.get_params()
parameters = {'alpha': (2,1, 0.1, 0.01, 0.001) }
clf = GridSearchCV(MultinomialNB(), parameters,refit=False,cv=2, n_jobs=-1)
clf.fit(x_train, y_train)
print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
print('Best Parameters : ',clf.best_params_)
Best Parameters :  {'alpha': 0.01}
'''

# Fitting XGBoost to the Training set
'''
from xgboost import XGBClassifier
model_xgboost = XGBClassifier()
train_model(model_xgboost,"XGBoost")
'''
'''
# Hyper Parameter Tunning
model_xgboost.get_params()
parameters = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.1, 0.5, 1],
        'subsample': [0.1, 0.3, 0.6],
        'colsample_bytree': [0.3, 0.6, 0.8],
        'max_depth': [1, 3, 4]
        }

clf = HalvingGridSearchCV(XGBClassifier(), parameters,refit=False,cv=2, n_jobs=-1)
clf.fit(x_train, y_train)
print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
print('Best Parameters : ',clf.best_params_)

Best Parameters :  {'colsample_bytree': 0.3, 'gamma': 0.1, 'max_depth': 3, 
                    'min_child_weight': 1, 'subsample': 0.6}
'''

# Fitting Gradient Boosting to the Training set
'''
from sklearn.ensemble import GradientBoostingClassifier
model_gradient_boosting = GradientBoostingClassifier()
train_model(model_gradient_boosting,"Gradient Boosting")
pickle.dump(model_gradient_boosting, open("satorp_GB.pkl", "wb"))
'''
'''
# Hyper Parameter Tunning
model_gradient_boosting.get_params()
parameters = {"learning_rate": [0.01,0.1,1,10,100],
              "max_depth":[1,3,5,7,9],
              "max_features":["log2","sqrt"],
              "subsample":[0.5, 0.75, 1.0],
              "n_estimators":[5,50,100,250]
              }

clf = GridSearchCV(GradientBoostingClassifier(), parameters,refit=False,cv=2, n_jobs=-1)
clf.fit(x_train, y_train)
print('Best Accuracy Through Grid Search : %.3f'%clf.best_score_)
print('Best Parameters : ',clf.best_params_)

Best Parameters :  
n_estimators=50, learning_rate=0.1,max_depth=1,max_features = 'log2', subsample = 0.5, random_state=0)
n_estimators=100, learning_rate=0.1,max_depth=5,max_features = 'sqrt', subsample = 1.0, random_state=0)

'''

# Export Output data
tstdf['Predicted Template']=y_pred
tstdf = tstdf.merge(y_pred_proba, left_index=True, right_index=True, how='inner')
tstdf.to_csv('output.csv')
model_accuracy_df.to_csv('Model_Accuracy.csv')

