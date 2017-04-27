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
data_simple['DOB_NA'] = np.where(data_simple['DOB'].isnull(),1,0)
data_simple['DOB'].fillna(mode, inplace=True)

#impute values for SCREENING_DISP_DATE as POLICE_RPT_DATE + SCREENING_DAYS
data_simple['SCREENING_DISP_DATE'] = np.where((data_simple['SCREENING_DISP_DATE'].isnull())&(data_simple['POLICE_RPT_DATE'].notnull()),\
                                         data_simple['POLICE_RPT_DATE'] + data_simple['SCREENING_DAYS'].apply(pd.offsets.Day),\
                                         data_simple['SCREENING_DISP_DATE'])

#impute values for BAR_ADMISSION as mode
mode = data_simple[data_simple['BAR_ADMISSION'].notnull()]['BAR_ADMISSION'].mode()[0]
data_simple['BAR_NA'] = np.where(data_simple['BAR_ADMISSION'].isnull(),1,0)
data_simple['BAR_ADMISSION'].fillna(mode, inplace=True)

#get year and month from each date variable
for col in date_features:
    data_simple[col+'_y'] = pe.get_year(data_simple[col])
    data_simple[col+'_m'] = pe.get_month(data_simple[col])
    


#drop rows with missing ARREST_DATE
data_simple = data_simple.loc[data_simple['ARREST_DATE'].notnull()]

#create age, days admitted to bar, days to screening decision variables
data_simple = data_simple.loc[data_simple['ARREST_DATE'].notnull()]
data_simple['AGE'] = data_simple['ARREST_DATE'] - data_simple['DOB']
data_simple['AGE'] = data_simple['AGE'].apply(lambda x: x.days)
#fill in mode for values where DOB was missing
data_simple['AGE'] = np.where(data_simple['DOB_NA']==1,data_simple[data_simple['DOB_NA']==0]['AGE'].mode()\
                              ,data_simple['AGE'])

data_simple['BAR_ADMIT_DAYS'] = data_simple['ARREST_DATE'] - data_simple['BAR_ADMISSION']
data_simple['BAR_ADMIT_DAYS'] = data_simple['BAR_ADMIT_DAYS'].apply(lambda x: x.days)
#fill in mode for values where DOB was missing
data_simple['BAR_ADMIT_DAYS'] = np.where(data_simple['BAR_NA']==1,data_simple[data_simple['BAR_NA']==0]['BAR_ADMIT_DAYS'].mode()\
                              ,data_simple['BAR_ADMIT_DAYS'])

data_simple['ARREST_TO_SCREEN'] = data_simple['SCREENING_DISP_DATE'] - data_simple['ARREST_DATE']
data_simple['ARREST_TO_SCREEN'] = data_simple['ARREST_TO_SCREEN'].apply(lambda x: x.days)
features_to_keep = ['UNIQUE_ID','POLICE_RPT_DATE','ARREST_DATE','DOB','SCREENING_DISP_DATE','BAR_ADMISSION',\
                   'POLICE_RPT_DATE_y','ARREST_DATE_y','DOB_y','SCREENING_DISP_DATE_y','BAR_ADMISSION_y',\
                   'POLICE_RPT_DATE_m','ARREST_DATE_m','DOB_m','SCREENING_DISP_DATE_m','BAR_ADMISSION_m',\
                   'AGE','BAR_ADMIT_DAYS','ARREST_TO_SCREEN']

df_date_features = data_simple[features_to_keep]

df_date_features.to_csv('df_date_features.csv', index=False)