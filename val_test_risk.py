import numpy as np
import pandas as pd
import project_env as pe
import sklearn as sklearn
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

train = pd.read_csv('train.csv')
val = pd.read_csv('val.csv')
test = pd.read_csv('test.csv')

#target variable of rearrest in 5 years
X_train, y_train = pe.create_target(train, years = 5)
X_val, y_val = pe.create_target(val, years = 5)
X_test, y_test = pe.create_target(test, years = 5)

#remove non-feature columns
X_train=X_train.drop(['BOFI_NBR','SCREENING_DISP_CODE','UNIQUE_ID','NEXT_ARREST_TIME'], axis=1)
X_val=X_val.drop(['BOFI_NBR','SCREENING_DISP_CODE','UNIQUE_ID','NEXT_ARREST_TIME'], axis=1)
X_test=X_test.drop(['BOFI_NBR','SCREENING_DISP_CODE','UNIQUE_ID','NEXT_ARREST_TIME'], axis=1)

#Using year and month as predictive variables
X_train=X_train.drop(['ARREST_DATE', 'ARREST_DATE_y'], axis=1)
X_val=X_val.drop(['ARREST_DATE','ARREST_DATE_y'], axis=1)
X_test=X_test.drop(['ARREST_DATE', 'ARREST_DATE_y'], axis=1)

# fit classifier
gbt = GradientBoostingClassifier(n_estimators=300,max_depth=5,min_samples_split=4)
gbt = gbt.fit(X_train,y_train)

# extract scores
val_risk = gbt.predict_proba(X_val)[:,1]
test_risk = gbt.predict_proba(X_test)[:,1]

# retrieve validation and test set with identifiers
X_val, y_val = pe.create_target(val, years = 5)
X_test, y_test = pe.create_target(test, years = 5)

# add risk to validation and test data, export
X_val['RISK']=val_risk
X_test['RISK']=test_risk
val_risk=X_val[['UNIQUE_ID','BOFI_NBR','RISK']]
test_risk=X_test[['UNIQUE_ID','BOFI_NBR','RISK']]
val_risk.to_csv('val_risk.csv',index=False)
test_risk.to_csv('test_risk.csv',index=False)