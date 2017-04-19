import pandas as pd
import numpy as np

# import data_simple
df = pd.read_csv('data_simple.csv',encoding = "ISO-8859-1")
print('original dataset shape: ',df.shape)

# convert values
# flags set to 1,0
# Sex set to 1,0 (indicates male, female, respectively)

df['CRIMINAL_FLAG'].loc[df['CRIMINAL_FLAG']=='N']=0
df['CRIMINAL_FLAG'].loc[df['CRIMINAL_FLAG']=='Y']=1

df['JUVENILE_FLAG'].loc[df['JUVENILE_FLAG']=='N']=0
df['JUVENILE_FLAG'].loc[df['JUVENILE_FLAG']=='Y']=1

df['FINAL_DETENTION_FLAG'].loc[pd.isnull(df['FINAL_DETENTION_FLAG'])]=0
df['FINAL_DETENTION_FLAG'].loc[df['FINAL_DETENTION_FLAG']=='N']=0
df['FINAL_DETENTION_FLAG'].loc[df['FINAL_DETENTION_FLAG']=='Y']=1

df['INITIAL_DETENTION_FLAG'].loc[pd.isnull(df['INITIAL_DETENTION_FLAG'])]=0
df['INITIAL_DETENTION_FLAG'].loc[df['INITIAL_DETENTION_FLAG']=='N']=0
df['INITIAL_DETENTION_FLAG'].loc[df['INITIAL_DETENTION_FLAG']=='Y']=1

df['SADA_SEX'].loc[np.logical_and(df['SADA_SEX']!='M',df['SADA_SEX']!='F')]=np.NaN
df['SADA_SEX'].loc[df['SADA_SEX']=='M']=1
df['SADA_SEX'].loc[df['SADA_SEX']=='F']=0

df['HABITUAL_OFFENDER_FLAG'].loc[np.logical_and(df['HABITUAL_OFFENDER_FLAG']!='Y',df['HABITUAL_OFFENDER_FLAG']!='N')]='N'
df['HABITUAL_OFFENDER_FLAG'].loc[df['HABITUAL_OFFENDER_FLAG']=='N']=0
df['HABITUAL_OFFENDER_FLAG'].loc[df['HABITUAL_OFFENDER_FLAG']=='Y']=1

df['SEX'].loc[np.logical_and(df['SEX']!='M',df['SEX']!='F')]=np.NaN
df['SEX'].loc[df['SEX']=='M']=1
df['SEX'].loc[df['SEX']=='F']=0

# select binary features
clean_data = df[['UNIQUE_ID','CRIMINAL_FLAG','FINAL_DETENTION_FLAG', \
                 'HABITUAL_OFFENDER_FLAG','INITIAL_DETENTION_FLAG', \
                 'JUVENILE_FLAG','SADA_SEX','SEX']]

# export
clean_data.to_csv('df_bin_features.csv',index=False)
print('new dataset shape: ',clean_data.shape)