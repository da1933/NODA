import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

def to_date(col):
    '''Given a pandas series, returns pandas dates for non-missing valid dates'''
    col.dropna(inplace=True)
    col = col[col!=0]
    col = col.astype(int)
    converted = pd.to_datetime(col, format='%Y%m%d', errors ='coerce')
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
    
def split_data(data, test_split=.2,  train_split=.64, by_var=None, random_state=None):
    '''
    Takes a dataframe and returns 3 data frames split into test, train, and validation.
    by_var takes a column name as an input and splits the data with an even distirbution of the values of that column
    The by_var colummn must NOT have any missing values and must have >1 row for each unique value.
    '''
    if by_var == None:
        temp, data_test = train_test_split(data, test_size = test_split, random_state = random_state)
        data_train, data_val = train_test_split(temp, test_size = 1-train_split/(1-test_split), random_state = random_state)
        return data_test, data_train, data_val
    else:
        sss_test = StratifiedShuffleSplit(n_splits=1, test_size = test_split, random_state = random_state)
        sss_test.get_n_splits(data,data[by_var])
        for train_index, test_index in sss_test.split(data,data[by_var]):
            temp, data_test = data.iloc[train_index], data.iloc[test_index]
            
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size = 1-train_split/(1-test_split), random_state = random_state)
        sss_val.get_n_splits(temp,temp[by_var])
        print(temp.shape)
        for train_index, val_index in sss_val.split(temp,temp[by_var]):
            print('VAL Data', len(val_index))
            print('Train Data', len(train_index))
            data_train, data_val = temp.iloc[train_index], temp.iloc[val_index]
        return data_test, data_train, data_val
    
def cnt_not_in_range(data, col, start='19880101', end='19991231'):
    '''
    Takes a dataframe and datetime column and returns the number of entries not within the range.
    '''
    cnt = data[data[col]< pd.to_datetime(start,format='%Y%m%d')][col].count()+\
          data[data[col]> pd.to_datetime(end,format='%Y%m%d')][col].count()
    return cnt

def get_year(col):
    '''Given a pandas series with datetime values, returns year for non-missing valid dates'''
    return col.map(lambda x: x.year)
        
def get_month(col):
    '''Given a pandas series with datetime values, returns year for non-missing valid dates'''
    return col.map(lambda x: x.month)
        