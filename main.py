from fastapi import FastAPI, File, UploadFile,Form
from io import BytesIO
from typing import Union
from pydantic import BaseModel
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from mlflow.tracking import MlflowClient

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support
LE = LabelEncoder()
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pickle
from sklearn.pipeline import make_pipeline
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
from fastapi.encoders import jsonable_encoder

warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.metrics import f1_score,precision_score,recall_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import RepeatedStratifiedKFold,KFold
from tpot import TPOTClassifier
import mysql.connector
from pydantic import BaseModel
from typing import Any, Dict, AnyStr, List, Union

class Item(BaseModel):
    Plant_name: str
    data: Union[List, Dict] = None


app = FastAPI()


### Data base connect 
def dbconnect():
    mydb = mysql.connector.connect(host="localhost",
    user="root",
    password="admin",
    database="loop_template_plant",
    port = '3306'
    )
    return mydb


### feature name fetch from database based on plant name
def fetchdata(mydb,plant_name):
    mycursor = mydb.cursor()
    sql = "SELECT feature_name,is_dependent FROM Data WHERE plant_name = %s"
    adr = (plant_name, )
    mycursor.execute(sql, adr)
    myresult = mycursor.fetchall()
    return myresult

### create the list of the features for independent varable and dependent variable 
def feature(data):
    feature_list =[]
    dependent_feature = []
    for i in data:
        if i[1] == 'TRUE':
            dependent_feature.append(i[0])
        else:
           feature_list.append(i[0])
    return feature_list,dependent_feature 

def fetchtraintestdata(plant_name,database_connect):
    if plant_name == 'Satorp' or plant_name == 'satorp' or plant_name == 'SATORP':
        trdf = pd.read_sql('SELECT * FROM satorp_training', con=database_connect)
        tstdf = pd.read_sql('SELECT * FROM satorp_test', con=database_connect)
    elif plant_name == 'J3' or plant_name == 'j3':
        trdf = pd.read_sql('SELECT * FROM J3_training', con=database_connect)
        trdf = pd.read_sql('SELECT * FROM J3_training', con=database_connect)
    elif plant_name == 'Dangote' or plant_name == 'dangote' or plant_name == 'DANGOTE':
        trdf = pd.read_sql('SELECT * FROM J3_training', con=database_connect)
        tstdf = pd.read_sql('SELECT * FROM J3_training', con=database_connect)
    else:
        return "Enter correct plant"
        print("Schema")
    return trdf,tstdf

### fetching the accuary for gobal model
def fetch_model_details(plant_name,database_connect):
    is_main = True
    mycursor = database_connect.cursor()
    sql ="select model_path,model_accurary from  model_Details where plant_name = %s and is_main = %s"
    adr = (plant_name,True )
    mycursor.execute(sql, adr)
    myresult = mycursor.fetchall()
    return myresult[0][0],myresult[0][1]

def updatetrainingdata(mydb,df):
    df.to_sql(con=mydb, name='satorp_training', if_exists='replace', flavor='mysql')

def updatetestdata(mydb,df):
    df.to_sql(con=mydb, name='satorp_test', if_exists='replace', flavor='mysql')
    



def group_tags(tag_db,input_feature_list,dependent_feature):
    for feature in input_feature_list:
        tag_db = pd.merge(tag_db, 
                          tag_db.groupby('LoopName')[feature].apply(list).reset_index(),
                             how='inner', on='LoopName')
        tag_db[feature+ '_y'] = ['' if set(l) == {''} else ','.join(map(str, l)) for l in tag_db[feature+ '_y']]
    tag_db = tag_db[['LoopName']+[s + '_y'  for s in input_feature_list]+dependent_feature]
    tag_db.columns = ['LoopName']+input_feature_list+dependent_feature
    return tag_db

#### Calling thr Retraining APi 
@app.post("/retrain")
async def retrain(item:Item):
    plant_name = item.Plant_name
    #return item
    #print(item.Plant_name)
    #print(plant_name)
    #print(changed_data['data'][0])
    #print(item.data[0])
    
    feature_list = []
    dependent_feature = []
    ### plant name test data path
    json_compatible_item_data = jsonable_encoder(item.data)
    tag_data = pd.DataFrame.from_dict(json_compatible_item_data)    
    #print(tag_data.shape)
    tag_data['Sig_Identifier_Group'] = tag_data['Tag'].str.split(
        '-', expand=True)[1]
    tag_data.rename(columns = {'System':'SYSTEM_Mod','IOType':'INPUT SIGL TYPE_Mod'}, inplace = True)
    my_db = dbconnect()
    data = fetchdata(my_db,plant_name)
    #print(data)
    feature_list,dependent_feature = feature(data)
    #print(feature_list)
    #print(dependent_feature)
    trdf,tstdf = fetchtraintestdata(plant_name,my_db)
    changed_data = group_tags(tag_data,feature_list,dependent_feature)
    #print(changed_data.columns)

    X_train = trdf[feature_list]
    changed_data_X= changed_data[feature_list]
    frames = [X_train, changed_data_X]
    #print(X_train.shape)
    X_train_new  =  pd.concat(frames)
    print(X_train_new.shape)
    y_train = trdf[dependent_feature]
    changed_data_Y =  changed_data[dependent_feature]
    frames = [y_train, changed_data_Y]
    y_train_new = pd.concat(frames)
    print(y_train_new.shape)
    X_test = tstdf[feature_list]
    frames = [X_test,changed_data_X]
    X_test_new = pd.concat(frames)
    y_test = tstdf[dependent_feature]
    frames = [y_test,changed_data_Y]
    y_test_new = pd.concat(frames)
    encode = Custom_encoder_data(feature_list)
    encode_old = Custom_encoder_data(feature_list)
    X_train = encode_old.fit(X_train)

    X_train_new = encode.fit(X_train_new)
    print("Len of uniques")
    print(len(encode.uniques))
    dump(encode, open('encode1.pkl', 'wb'))
    X_test_old = encode_old.transform(X_test_new)
    X_test_new = encode.transform(X_test_new)
    X_test_comp = encode.transform(X_test)

    cv = KFold(n_splits=10,shuffle=True, random_state=1)
    client = MlflowClient()
    exps = mlflow.list_experiments()
    flag = False
    for exp in exps:
        if exp.name == plant_name:
            mlflow.set_experiment(plant_name)

            flag =True
            break

    if flag == False:
        experiment_id = client.create_experiment(plant_name)
        mlflow.set_experiment(plant_name)

        


   


    mlflow.start_run()
    acc_old_with_new_data = '96.26'
    model = TPOTClassifier(generations=6, population_size=50, cv=cv, scoring='accuracy', verbosity=2, random_state=1, n_jobs=-1,warm_start = True)
    model.fit(X_train_new,y_train_new)
    print("best model :",end=" ")
    print(model.fitted_pipeline_)
    y_prediction =model.fitted_pipeline_.predict(X_test_new)
    print('Accuary:',model.fitted_pipeline_.score(X_test_new,y_test_new))
    #print(dir(model.fitted_pipeline_))
    print('f1_score:',f1_score(y_prediction,y_test_new,average='weighted'))
    print('Precision Score:',precision_score(y_prediction,y_test_new,average='weighted'))
    print('Recall:',recall_score(y_prediction,y_test_new,average='weighted'))
    print(model.fitted_pipeline_._estimator_type)
    print(model.fitted_pipeline_._final_estimator)
    print(model.fitted_pipeline_._final_estimator.get_params())
    mlflow.log_param("Model_details ", str(model.fitted_pipeline_._final_estimator).split('(')[0])
    for i in model.fitted_pipeline_._final_estimator.get_params():
        mlflow.log_param(i,model.fitted_pipeline_._final_estimator.get_params()[i])
    accuracy = model.fitted_pipeline_.score(X_test_new,y_test_new)
    accuracy_comp  = model.fitted_pipeline_.score(X_test_comp,y_test)
    print(accuracy_comp)
    mlflow.log_metric('Accuary',accuracy)
    F1_Score = f1_score(y_prediction,y_test_new,average='weighted')
    mlflow.log_metric('f1_score',F1_Score)
    #mlflow.log_metric('Precision Score:',precision_score(y_prediction,y_test,average='weighted'))
    #mlflow.log_metric('Recall:',recall_score(y_prediction,y_test,average='weighted'))
    gobal_model_path,gobal_model_acc = fetch_model_details(plant_name,my_db)
    c= load(open(gobal_model_path, "rb"))
    #y_pred_old = c.predict(X_test_old)
    #c.fit()
    print(acc_old_with_new_data)
    accuary = model.fitted_pipeline_.score(X_test_new,y_test_new)
    if accuracy_comp >= (gobal_model_acc-2)/100:
        if accuary >0.9:
            dump(model.fitted_pipeline_, open('models/Satorp/gobal/modelv.pkl', 'wb'))
            ### update the training dataset
            data_changed =  pd.concat([changed_data_X,changed_data_Y], axis=1)
            print(data_changed.head())
            ### update the test dataset
        else:
            return {"Error": "offline analysis"}
    else:
        dump(model.fitted_pipeline_, open('models/Satorp/local/model.pkl', 'wb'))



    #dump(model.fitted_pipeline_, open('model.pkl', 'wb'))
    #mlflow.log_artifact('model.pkl')
    mlflow.end_run()
    return {"Accuary": model.fitted_pipeline_.score(X_test_new,y_test_new)*100,'plant_name':plant_name,'Gobal_Accuary':gobal_model_acc}



