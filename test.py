from pickle import dump
from pickle import load
import pandas as pd
from collections import Counter
from sklearn.metrics import accuracy_score
#from encode import Custom_encoder_data
import pandas as pd

### path of tranining data and test data 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
#train_file = r"training_data_unique.csv"
test_file = r"test_data.csv"
feature_list = ['INPUT SIGL TYPE_Mod', 'Sig_Identifier_Group','InstrumentType','SYSTEM_Mod', 'SubSystem']
feture_name = {'stn':"StrategyTemplateName_Final",'io': "INPUT SIGL TYPE_Mod",'sigi':"Sig_Identifier_Group",'syst':"SYSTEM_Mod",'subsyst':"SubSystem",'InstrumentType':"InstrumentType"}

dependent_feature = ['StrategyTemplateName_Final']

###   loading the traning data and testing data
tstdf = pd.read_csv(test_file)
#tstdf = tstdf[tstdf['StrategyTemplateName_Final'] != '$DCS_DI_ALM']
X_test = tstdf[feature_list]
y_test = tstdf[dependent_feature]
#clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
### 
encod = load((open('encode1.pkl', "rb")))
#print(e)
X_test_new = encod.transform(X_test)
print(X_test_new.shape)
model = load((open('model.pkl', "rb")))
#model.fit(X_train_new,y_train.values.ravel())
print('Accuracy is :',accuracy_score(model.predict(X_test_new),y_test))