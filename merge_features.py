import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import project_env as pe
from sklearn.preprocessing import LabelEncoder

#bring in BOFI_NBR, SCREENING_DISP_CODE
data_simple = pd.read_csv('data_simple.csv', encoding = "ISO-8859-1", low_memory=False)
data_simple = data_simple[['SCREENING_DISP_CODE','UNIQUE_ID','BOFI_NBR']]

bin_features = pd.read_csv('df_bin_features.csv', encoding = "ISO-8859-1", low_memory=False)
num_features = pd.read_csv('df_num_features.csv', encoding = "ISO-8859-1", low_memory=False)
date_features = pd.read_csv('df_date_features.csv', encoding = "ISO-8859-1", low_memory=False)\
			.drop('JUVENILE_FLAG',axis=1)
cat_features = pd.read_csv('df_cat_features.csv', encoding = "ISO-8859-1", low_memory=False)
rearrest = pd.read_csv('df_rearrest_times.csv', encoding = "ISO-8859-1", low_memory=False)

merged = pd.merge(rearrest, \
                 bin_features,\
                 on='UNIQUE_ID', \
                 how='left')
merged = pd.merge(merged, \
                 num_features,\
                 on='UNIQUE_ID', \
                 how='left')
merged = pd.merge(merged, \
                 date_features,\
                 on='UNIQUE_ID', \
                 how='left')
merged = pd.merge(merged, \
                 cat_features,\
                 on='UNIQUE_ID', \
                 how='left')
merged = pd.merge(merged, \
                 data_simple,\
                 on='UNIQUE_ID', \
                 how='left')

merged = merged[['UNIQUE_ID', 'NEXT_ARREST_TIME', 'ARREST_DATE', \
	'ARREST_DATE_y','BOFI_NBR','SCREENING_DISP_CODE',\
        'BAR_ADMIT_DAYS','CRIMINAL_FLAG', \
        'FINAL_DETENTION_FLAG', 'HABITUAL_OFFENDER_FLAG', \
        'INITIAL_DETENTION_FLAG', 'JUVENILE_FLAG', 'SADA_SEX', \
        'SEX', 'TOT_NUM_DEF', 'MULTIPLE_DEF_FLAG', 'SCREENING_DAYS', \
        'SCREENING_DISP_DATE_y', \
        'SCREENING_DISP_DATE_m', 'AGE', 'ARREST_TO_SCREEN', \
        'CHARGE_CLASS', 'CHARGE_TYPE', 'PARTY', 'RACE', \
        'SADA_RACE']]

merged.to_csv('merged.csv',index=False)
print('Exported merged.csv\n')


#identify arrests where at least one charge was accepted
accepted = merged[merged['SCREENING_DISP_CODE']==230][['BOFI_NBR','ARREST_DATE']]

#drop rows where at least one charge was accepted during that arrest
refused = pd.merge(merged, \
                 accepted, \
                 on=['BOFI_NBR','ARREST_DATE'], \
                 how='outer',\
                 indicator = True)

refused = refused[refused['_merge']=='left_only'].drop('_merge', axis=1)

#remove duplicate arrests on same day
refused = refused[refused['NEXT_ARREST_TIME']!='Delete']

#convert NEXT_ARREST_TIME to numeric
refused['NEXT_ARREST_TIME'] = refused['NEXT_ARREST_TIME'].apply(pd.to_numeric)

# Split and export data with non-encoded categorical variables
test_ne, train_ne, val_ne = pe.split_data(refused, test_split=.2, \
                            train_split=.64, by_var='ARREST_DATE_y', random_state=1)
train_ne.to_csv('data_train.csv', index=False)
test_ne.to_csv('data_test.csv', index=False)
val_ne.to_csv('data_val.csv', index=False)
print('Exported non-encoded, split data\n')


# Encode categorical variables
cat_var = ['SADA_SEX', 'SEX', 'PARTY', 'RACE', 'SADA_RACE', \
           'SCREENING_DISP_DATE_y', 'SCREENING_DISP_DATE_m', 'CHARGE_TYPE', \
           'CHARGE_CLASS', 'ARREST_DATE_y']

cat_var_enc = pe.one_hot_encode(refused[cat_var])
cat_var_enc = pd.DataFrame(cat_var_enc.toarray(), index=refused.index)
refused_enc = refused.drop(cat_var[:-1], axis=1)
refused_enc = pd.merge(refused_enc, \
                       cat_var_enc,\
                       left_index=True, \
                       right_index=True, \
                       how='left')

# Split and export encoded data
test, train, val = pe.split_data(refused_enc, test_split=.2, \
                   train_split=.64, by_var='ARREST_DATE_y', random_state=1)
train.to_csv('train.csv',index=False)
val.to_csv('val.csv',index=False)
test.to_csv('test.csv', index=False)
print('Exported encoded, split data\n')

#######################################
# Features for Baseline Decision Tree #
#######################################

refused_dt = refused[['UNIQUE_ID','CHARGE_CLASS','AGE','ARREST_DATE_y','NEXT_ARREST_TIME']]
refused_dt = refused_dt.dropna(axis=0,subset=['CHARGE_CLASS'])

#for visualization purposes for decision tree, age converted to years
refused_dt['AGE'] = refused_dt['AGE']/365.0

# Split and export baseline decision tree data
test_dt, train_dt, val_dt = pe.split_data(refused_dt, test_split=.2, \
                                 train_split=.64, by_var='ARREST_DATE_y', random_state=1)
train_dt.to_csv('train_dt.csv',index=False)
val_dt.to_csv('val_dt.csv',index=False)
test_dt.to_csv('test_dt.csv', index=False)
print('Exported baseline decision tree data')


















