# NOTE: This data merge requires that all source CSV files
# are in a folder named "csv" that resides in the same 
# directory as this .py file. Individual CSV files names 
# have not been changed from their version in the  
# original data collection

# The output of this script will be a database called 
# data_simple.csv that can be used for the prediction model

import numpy as np
import pandas as pd

#############################################################
# Rename ADA attributes to distinguish them from defendants #
#############################################################

old_names = ['DOB', 'RACE', 'SEX']

new_names = ['SADA_DOB', 'SADA_RACE', 'SADA_SEX']

name_dict=dict(zip(old_names, new_names))


##########################################
# Import and process relevant data files #
##########################################
# Defendant summary related to charges
dsum = pd.read_table("source/Dsum-cln.csv", sep = '^', \
		     dtype='object', index_col=False)

# Arrest registry
areg = pd.read_table("source/Areg-cln.csv", sep = '^', \
		     dtype='object', index_col=False)

# District Attorney information
ada  = pd.read_table("source/Ada-cln.csv", sep = '^', \
		     dtype='object', index_col=False)

# Defendant history
dfdn = pd.read_table("source/Dfdn-cln.csv", sep = '^', \
             dtype='object', index_col=False) \
		     .sort_values(['BOFI_NBR','ADDR_1']) \
		     .drop_duplicates('BOFI_NBR')


##################################
# Select attrtibutes of interest #
##################################

dsum_cln = dsum[['ADA_CODE', 'BOFI_NBR', 'DFDN_SEQ_NBR', \
		 'SCREENING_DISP_CODE', 'SYS_NBR', \
		 'POLICE_RPT_DATE', 'POLICE_RPT_DAYS', \
		 'SCREENING_DAYS', 'SCREENING_DISP_DATE']]

areg_cln = areg[['ARREST_CREDIT_CODE', \
		 'ARREST_DATE', 'ADD_DATE', 'BOFI_NBR', \
		 'SYS_NBR', 'CHARGE_CLASS', 'CHARGE_TYPE', \
		 'DFDN_SEQ_NBR', 'HABITUAL_OFFENDER_FLAG', \
		 'FINAL_DETENTION_FLAG', 'INITIAL_DETENTION_FLAG', \
		 'LEAD_CHARGE_CODE']]

ada_cln  = ada[['ADA_CODE', 'BAR_ADMISSION', 'DOB', \
		'RACE', 'SEX', 'PARTY']]

dfdn_cln = dfdn[['BOFI_NBR', 'JUVENILE_FLAG', 'CRIMINAL_FLAG', \
		 'FBI_NBR', 'DOB', 'SEX', 'RACE']]


######################################
# Merge all data into one data frame #
######################################

data_merged = pd.merge(dsum_cln, areg_cln, \
		on=['BOFI_NBR', 'DFDN_SEQ_NBR', \
		    'SYS_NBR'], how='left')


#New ADA names applied here
data_merged = pd.merge(data_merged, ada_cln, \
		on='ADA_CODE', how='left') \
		.rename(columns=name_dict)

data_simple = pd.merge(data_merged, dfdn_cln, \
		on='BOFI_NBR', how='inner')

###########################################
# Add ID column, sort, and export to file #
###########################################
data_simple['UNIQUE_ID'] = pd.Series(np.arange(data_simple.shape[0]))
cols = list(np.sort(data_simple.columns.values))
cols.insert(0,(cols.pop()))
data_simple = data_simple.ix[:,cols]
data_simple.to_csv('output/data_simple.csv', index=False)