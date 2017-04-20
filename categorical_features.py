import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

merged_data = pd.read_csv('data_simple.csv', encoding = "ISO-8859-1", low_memory=False)
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

CGCD = pd.read_stata("cgcd-cln.dta")
Code = pd.read_stata("code-cln.dta")

merged_cat_filled = pd.merge(merged_cat_filled, CGCD[['charge_code', 'charge_class', 'charge_desc']], \
                             left_on=['LEAD_CHARGE_CODE'], right_on=['charge_code'], how='left')

#Change Lead Charge Code to NA if not found in CGCD
unmatched = merged_cat_filled.loc[merged_cat_filled['charge_code'].isnull() == True,'LEAD_CHARGE_CODE'].unique()
merged_cat_filled.loc[merged_cat_filled['LEAD_CHARGE_CODE'].isin(unmatched),'LEAD_CHARGE_CODE'] = 'NA'

merged_cat_filled.drop(['charge_code','charge_class','charge_desc'], axis = 1).to_csv('df_cat_features.csv',index=False)

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

categorical_data_dummies.to_csv('df_cat_features.csv',index=False)
'''



