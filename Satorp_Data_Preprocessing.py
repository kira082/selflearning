# -*- coding: utf-8 -*-

# importing libraries
import pandas as pd
import numpy as np
from collections import Counter


def process_satorp_data(tag_data):
    # Remove rows where loop name is null
    tag_data = tag_data[~tag_data.LoopName.isnull()]

    # Filter data with system as DCS and non-serial IOTypes
    tag_data = tag_data[tag_data.System == 'DCS']
    tag_data = tag_data[~tag_data.IOType.isin(
        ['CAI', 'CAO', 'CDI', 'CDO', 'S', np.NAN])]

    tag_data['Tag_Modified'] = tag_data['Tag'].str.split(
        '-', expand=True).iloc[:, 1:].fillna("").agg("".join, axis=1)

    # Removing special characters from Loop name
    tag_data["Loop_Modified"] = tag_data['LoopName'].str.replace('-', '')

    # Extracting "Signal_Identifier" column from Tag_name
    tag_data['Sig_Identifier_Group'] = tag_data['Tag'].str.split(
        '-', expand=True)[1]

    satorp_data = tag_data.copy() # Creating a copy of DataFrame to pass to block level prediction
    # Grouping the dataframe by loop name
    merged_df = pd.merge(satorp_data,
                         satorp_data.groupby('Loop_Modified')[
                             'InstrumentType'].apply(list).reset_index(),
                         how='inner', on='Loop_Modified')
    merged_df = pd.merge(merged_df,
                         merged_df.groupby('Loop_Modified')[
                             'System'].apply(list).reset_index(),
                         how='inner', on='Loop_Modified')
    merged_df = pd.merge(merged_df,
                         merged_df.groupby('Loop_Modified')[
                             'SubSystem'].apply(list).reset_index(),
                         how='inner', on='Loop_Modified')
    merged_df = pd.merge(merged_df,
                         merged_df.groupby('Loop_Modified')[
                             'IOType'].apply(list).reset_index(),
                         how='inner', on='Loop_Modified')
    merged_df = pd.merge(merged_df,
                         merged_df.groupby('Loop_Modified')[
                             'Tag'].apply(list).reset_index(),
                         how='inner', on='Loop_Modified')
    merged_df = pd.merge(merged_df,
                         merged_df.groupby('Loop_Modified')[
                             'Sig_Identifier_Group'].apply(list).reset_index(),
                         how='inner', on='Loop_Modified')

    merged_df = merged_df[['UnitName', 'PlantName', 'LoopName', 'InstrumentType_x',
                           'Tag_y', 'InstrumentType_y', 'System_y', 'SubSystem_y', 'IOType_y', 'Sig_Identifier_Group_y']]
    merged_df.columns = ['UnitName', 'PlantName', 'LoopName', 'Loop_Type',
                         'Tag_Modified_Group', 'InstrumentType', 'SYSTEM_Mod', 'SubSystem', 'INPUT SIGL TYPE_Mod', 'Sig_Identifier_Group']

    training_data = merged_df[['LoopName', 'INPUT SIGL TYPE_Mod',
                               'Sig_Identifier_Group', 'InstrumentType', 'SYSTEM_Mod', 'SubSystem']]

    training_data['INPUT SIGL TYPE_Mod'] = ['' if set(l) == {''} else ','.join(
        map(str, l)) for l in training_data['INPUT SIGL TYPE_Mod']]
    training_data['Sig_Identifier_Group'] = ['' if set(l) == {''} else ','.join(
        map(str, l)) for l in training_data['Sig_Identifier_Group']]
    training_data['InstrumentType'] = ['' if set(l) == {''} else ','.join(
        map(str, l)) for l in training_data['InstrumentType']]
    training_data['SYSTEM_Mod'] = ['' if set(l) == {''} else ','.join(
        map(str, l)) for l in training_data['SYSTEM_Mod']]
    training_data['SubSystem'] = ['' if set(l) == {''} else ','.join(
        map(str, l)) for l in training_data['SubSystem']]
    return training_data


def process_satorp_Block(tag_data):
    tag_data.rename(columns = {'System':'SYSTEM_Mod','IOType':'INPUT SIGL TYPE_Mod'}, inplace = True)
    block_df = tag_data[['INPUT SIGL TYPE_Mod','Sig_Identifier_Group','InstrumentType', 'SYSTEM_Mod','SubSystem','State1','Strategy_Template']]
    encoded_data = encode_features(block_df)
    return encoded_data


def encode_features(df):
    # Define features used for training
    stn = "Strategy_Template"
    io = "INPUT SIGL TYPE_Mod"
    sigi = "Sig_Identifier_Group"
    syst = "SYSTEM_Mod"
    State1 = "State1"
    # Encode Categorical features
    cols_to_encode = [io, sigi, syst, State1, stn]
   # block_df[pd.isnull(block_df)]  = ''
    uniques = {k for c in cols_to_encode for i in list(
        df[c].unique()) for k in str(i).lower().split(",")}
    uniques = sorted(uniques)

    def f():
        return {i: 0 for i in uniques}
    rows = []
    for i, r in df.iterrows():
        nr = f()
        #nr[Base_Block] = r[Base_Block]
        d = Counter(
            [k for c in cols_to_encode for k in str(r[c]).lower().split(',')])
        nr.update(d)
        d = [nr[k] for k in uniques]
        rows.append(d)
    return pd.DataFrame(rows, columns=uniques)
