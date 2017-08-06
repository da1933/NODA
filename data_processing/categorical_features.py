import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

merged_data = pd.read_csv('output/data_simple.csv', encoding = "ISO-8859-1", low_memory=False)
categorical = ['UNIQUE_ID', 'ARREST_CREDIT_CODE', 'CHARGE_CLASS', 'CHARGE_TYPE', 'LEAD_CHARGE_CODE',\
               'PARTY', 'RACE', 'SADA_RACE']
merged_cat_filled = merged_data[categorical].fillna('NA')
invalid_charge_codes = ['40:(979)296', 
               '14:(24)30(',
               '14:(24)67(',
               '5:606',
               '13:34',
               'F5:257',
               '14:(26)67(',
               '40:(979)1967',
               '4:664']

merged_cat_filled['LEAD_CHARGE_CODE'].replace(invalid_charge_codes, 'NA', inplace=True)

merged_cat_filled['RACE'].replace('N', 'B', inplace=True)
merged_cat_filled['RACE'].replace('O','A', inplace=True)

merged_cat_filled['ARREST_CREDIT_CODE'].replace('00', 'NA', inplace=True)

CGCD = pd.read_stata("source/cgcd-cln.dta")
Code = pd.read_stata("source/code-cln.dta")

#Convert class from number to description

Class_names = Code[Code.code_type == 'CSCLCD']
Class_names = Class_names[Class_names.code_code != '']
Class_names.code_code = Class_names.code_code.astype(int)

merged_cat_filled = pd.merge(merged_cat_filled, Class_names[['code_code', 'long_desc']], \
        left_on='CHARGE_CLASS', right_on='code_code')

merged_cat_filled.drop(['CHARGE_CLASS', 'code_code'], axis = 1, inplace=True)
merged_cat_filled.rename(columns={"long_desc": "CHARGE_CLASS"}, inplace=True)

merged_cat_filled.CHARGE_CLASS.replace('CASE CLASS CODES', 'NA', inplace=True)

#Join in Charge Descriptions

merged_cat_filled = pd.merge(merged_cat_filled, CGCD[['charge_code', 'charge_class', 'charge_desc']], \
                             left_on=['LEAD_CHARGE_CODE'], right_on=['charge_code'], how='left')

#Change Lead Charge Code to NA if not found in CGCD
unmatched = merged_cat_filled.loc[merged_cat_filled['charge_code'].isnull() == True,'LEAD_CHARGE_CODE'].unique()
merged_cat_filled.loc[merged_cat_filled['LEAD_CHARGE_CODE'].isin(unmatched),'LEAD_CHARGE_CODE'] = 'NA'

#Change Charge Desc to NA if it doesn't occur at least 100 times
charge_desc = pd.DataFrame(merged_cat_filled.charge_desc.value_counts())
rare_charge_descs = list(charge_desc[charge_desc.charge_desc <= 100].index)
merged_cat_filled.loc[merged_cat_filled['charge_desc'].isin(rare_charge_descs),'charge_desc'] = 'NA'

#Capitalize Charge Desc
merged_cat_filled.rename(columns={"charge_desc": "CHARGE_DESC"}, inplace = True)

#Altered this to not drop the charge_desc
merged_cat_filled.drop(['charge_code','charge_class'], axis = 1).to_csv('output/df_cat_features.csv',index=False)

#This is also where I should make the charge class a name instead of a number for easier model interpretation

'''
create features using one hot encoding
'''
'''
# first convert into integer values. one hot enconding only takes int input

l_enc = LabelEncoder()

col1 = l_enc.fit_transform(merged_cat_filled['CHARGE_CLASS'].astype(str))
col2 = l_enc.fit_transform(merged_cat_filled['CHARGE_TYPE'])
col3 = l_enc.fit_transform(merged_cat_filled['LEAD_CHARGE_CODE'])
col4 = l_enc.fit_transform(merged_cat_filled['PARTY'])
col5 = l_enc.fit_transform(merged_cat_filled['RACE'])
col6 = l_enc.fit_transform(merged_cat_filled['SADA_RACE'])
col7 = l_enc.fit_transform(merged_cat_filled['ARREST_CREDIT_CODE'])

X = np.column_stack((col1,col2,col3,col4,col5,col6,col7))

#one hot encoding
enc = OneHotEncoder()
X_enc = enc.fit_transform(X)

#combining sparse matrix with the unique ID

categorical_data_dummies = pd.DataFrame(merged_cat_filled['UNIQUE_ID'])
df = pd.DataFrame(X_enc.toarray())
categorical_data_dummies = pd.concat([df, categorical_data_dummies], axis = 1)

categorical_data_dummies.to_csv('output/df_cat_features.csv',index=False)
'''



