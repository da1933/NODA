import numpy as np
import pandas as pd

def to_date(col):
    '''Given a pandas series, returns pandas dates for non-missing valid dates'''
    col.dropna(inplace=True)
    col = col[col!=0]
    col = col.astype(int)
    converted = pd.to_datetime(col, errors='coerce', format='%Y%m%d')
    return converted
    

def getDfSummary(input_data):
    summary = input_data.describe().transpose()
    summary = summary.drop(['count'], axis=1)

    number_distinct = input_data.unstack().groupby(level=0).nunique(dropna=True)
    number_distinct.name = "number_distinct"

    number_nan = input_data.isnull().sum()
    number_nan.name = "number_nan"
    
    data_type = input_data.dtypes
    data_type.name = "type"
    
    count = input_data.count()
    count.name='valid_count'

    output_data = pd.concat([data_type,summary, number_distinct, count, number_nan], axis=1)
   
    return output_data
