import os
import pandas as pd
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from pickle import dump
stn = "StrategyTemplateName_GRP"
io = "INPUT SIGL TYPE_Mod"
sigi = "Sig_Identifier_Group"
syst = "SYSTEM_Mod"
#subsyst = "SubSystem"
#InstrumentType = 'InstrumentType'
feature_name =  {'stn':"StrategyTemplateName_Final",'io': "INPUT SIGL TYPE_Mod",'sigi':"Sig_Identifier_Group",'syst':"SYSTEM_Mod",'subsyst':"SubSystem",'InstrumentType':"InstrumentType"}
LE = LabelEncoder()
train_file = r"training_data_unique.csv"
#test_file = r"test_data.csv"
feature_list = ['INPUT SIGL TYPE_Mod', 'Sig_Identifier_Group','InstrumentType','SYSTEM_Mod', 'SubSystem']
dependent_feature = ['StrategyTemplateName_Final']
trdf = pd.read_csv(train_file)
#tstdf = pd.read_csv(test_file)
trdf = trdf[feature_list+dependent_feature]
#test_df = tstdf[feature_list+dependent_feature]
cols_to_encode = feature_list
trdf[pd.isnull(trdf)]  = ''
uniques = {k for c in cols_to_encode for i in list(trdf[c].unique()) for k in str(i).lower().split(",")}
uniques = sorted(uniques)
uniques.append(dependent_feature[0])
stn = feature_name['stn']
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
preprocess_data = {'uniques':uniques,'feature_list':feature_list,'feature_name':feature_name}
dump(preprocess_data, open('encode.pkl', 'wb'))