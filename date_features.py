import numpy as np
import pandas as pd
import project_env as pe
from datetime import timedelta, datetime

data_simple = pd.read_csv('data_simple.csv', encoding = "ISO-8859-1", low_memory=False)

#convert dates to python dates
date_features = ['POLICE_RPT_DATE','ARREST_DATE','ADD_DATE','SADA_DOB','DOB','SCREENING_DISP_DATE','BAR_ADMISSION']

for feature in date_features:
    data_simple[feature] = pe.to_date(data_simple[feature])
    
#impute values for POLICE_RPT_DATE as ARREST_DATE + POLICE_RPT_DAYS
data_simple['POLICE_RPT_DATE'] = np.where((data_simple['POLICE_RPT_DATE'].isnull())&(data_simple['ARREST_DATE'].notnull()),\
                                         data_simple['ARREST_DATE'] + data_simple['POLICE_RPT_DAYS'].apply(pd.offsets.Day),\
                                         data_simple['POLICE_RPT_DATE'])

#impute values for DOB as mode
mode = data_simple[data_simple['DOB'].notnull()]['DOB'].mode()[0]
data_simple['DOB'].fillna(mode, inplace=True)

#impute values for SCREENING_DISP_DATE as POLICE_RPT_DATE + SCREENING_DAYS
data_simple['SCREENING_DISP_DATE'] = np.where((data_simple['SCREENING_DISP_DATE'].isnull())&(data_simple['POLICE_RPT_DATE'].notnull()),\
                                         data_simple['POLICE_RPT_DATE'] + data_simple['SCREENING_DAYS'].apply(pd.offsets.Day),\
                                         data_simple['SCREENING_DISP_DATE'])

#impute values for BAR_ADMISSION as mode
mode = data_simple[data_simple['BAR_ADMISSION'].notnull()]['BAR_ADMISSION'].mode()[0]
data_simple['BAR_ADMISSION'].fillna(mode, inplace=True)

#get year and month from each date variable
for col in date_features:
    data_simple[col+'_y'] = pe.get_year(data_simple[col])
    data_simple[col+'_m'] = pe.get_month(data_simple[col])
    
features_to_keep = ['UNIQUE_ID','POLICE_RPT_DATE','ARREST_DATE','DOB','SCREENING_DISP_DATE','BAR_ADMISSION',\
                   'POLICE_RPT_DATE_y','ARREST_DATE_y','DOB_y','SCREENING_DISP_DATE_y','BAR_ADMISSION_y',\
                   'POLICE_RPT_DATE_m','ARREST_DATE_m','DOB_m','SCREENING_DISP_DATE_m','BAR_ADMISSION_m']

df_date_features = data_simple[features_to_keep]

#drop rows with missing ARREST_DATE
df_date_features = df_date_features[df_date_features['ARREST_DATE'].notnull()]

df_date_features.to_csv('df_date_features.csv', index=False)