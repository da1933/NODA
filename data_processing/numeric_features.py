import numpy as np
import pandas as pd
import project_env as pe
from datetime import timedelta, datetime

data_simple = pd.read_csv('output/data_simple.csv', encoding = "ISO-8859-1", low_memory=False)

#Get total # of defendents for each case
data_simple['TOT_NUM_DEF'] = data_simple.groupby('SYS_NBR')['DFDN_SEQ_NBR'].transform('max')

#Indicator Variable for whether there were multiple defendents
data_simple['MULTIPLE_DEF_FLAG'] = np.where(data_simple['TOT_NUM_DEF'] > 1,1,0)

df_num_features = data_simple[['UNIQUE_ID','TOT_NUM_DEF','MULTIPLE_DEF_FLAG','SCREENING_DAYS','POLICE_RPT_DAYS']]
df_num_features.to_csv('output/df_num_features.csv', index=False)