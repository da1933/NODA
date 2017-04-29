import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import project_env as pe

class Error_Analysis():
    def __init__(self, filekey, x_file, y_file):
        self.filekey = filekey
        self.x_file = x_file
        self.y_file = y_file

    def create_err_analysis_df(self, years):
        '''Takes (1) a filename where the non-encoded x data has been saved
        (2) a filename where the results associated with the x data have been saved and
        (3) the number of years for the creation of the categorical target variable '''
        self.err_analysis = pd.read_csv(self.x_file, encoding = "ISO-8859-1", low_memory=False, index_col=0, )
        self.err_analysis, y = pe.create_target(self.err_analysis, years)
        self.err_analysis['True_Y'] = y.copy()
        pred_data = np.load(self.y_file)
        self.err_analysis['Pred_Y'] = pred_data.copy()
        self.err_analysis['CORRECT'] = np.where((self.err_analysis['True_Y'] == self.err_analysis['Pred_Y']), 1, 0)
    
    def divide_preds(self):
        self.wrong_pred = self.err_analysis[self.err_analysis['CORRECT'] == 0]
        self.right_pred = self.err_analysis[self.err_analysis['CORRECT'] == 1]
        
    def plot_boxplots(self, numeric_column_names):
        '''Takes (1) a list of the numeric column names
        (2) a dataframe of incorrectly predicted datapoints
        (3) a dataframe of correctly predicted data points and
        (4) a name for saving the plot'''
        plt.figure(1)
        plt.gcf().set_size_inches(12,30) 

        rows = len(numeric_column_names)

        i = 1

        for variable in numeric_column_names:
            ax1 = plt.subplot(rows,2,i)
            if i ==1:
                ax1.set_title('Incorrect Prediction')
            self.wrong_pred[variable].plot.box()
            i = i + 1
            ax2 = plt.subplot(rows,2,i, sharey=ax1)
            if i ==2:
                ax2.set_title('Correct Prediction')
            self.right_pred[variable].plot.box()
            i = i + 1
        plt.savefig(self.filekey + '_boxplots.png')
    
    def mean_comparison(self, numeric_column_names):
        '''Takes list of column names, outputs file of a comparison of the means within those columns'''
        mean_wrong_pred = self.wrong_pred.describe().loc['mean',]
        mean_right_pred = self.right_pred.describe().loc['mean',]
        mean_comparison = pd.concat([mean_wrong_pred, mean_right_pred], axis=1).loc[numeric_column_names,]
        mean_comparison.columns = ['Incorrectly Predicted', 'Correctly Predicted']
        mean_comparison.to_csv(self.filekey + '_mean_comparison.csv')
        
    def correlation(self, numeric_column_names):
        '''Takes a list of column names, outputs file of the correlations between those variables and the y-value'''
        corr = self.err_analysis[numeric_column_names].corrwith(self.err_analysis['True_Y'], )
        corr.to_csv(self.filekey + '_correlation_with_y.csv')
        
    def make_cat_breakdowns(self, categorical_column_names):
        '''Takes a list of categorical column names, makes a dictionary of the percentages for the proportion of each top
        category for the incorrectly and correctly predicted datapoints'''
        pct_cat_wrong = {}
        pct_cat_right = {}
        self.pct_cat_compare = {}

        for var in categorical_column_names:
            pct_cat_wrong[var] = self.wrong_pred[var].value_counts()/len(self.wrong_pred)
            pct_cat_right[var] = self.right_pred[var].value_counts()/len(self.right_pred)
            self.pct_cat_compare[var] = pd.concat([pd.DataFrame(pct_cat_wrong[var]), pd.DataFrame(pct_cat_right[var])], axis=1)
            self.pct_cat_compare[var].columns = ['Incorrectly Predicted', 'Correctly Predicted']

    def plot_cat_breakdowns(self):
        '''Creates a bar plot for each of the to 5 values of the categorical variables for incorrectly and correctly
        predicted datapoints'''
        
        cat_var = list(self.pct_cat_compare.keys())
        
        fig, axes = plt.subplots(nrows=len(cat_var), ncols=1)
        
        for i in range(len(cat_var)):
            self.pct_cat_compare[cat_var[i]].sort_values(by='Incorrectly Predicted', ascending=False)\
            .head().plot.bar(title=cat_var[i], rot=0, ax=axes[i])

        plt.gcf().set_size_inches(12,30) 
        plt.gcf().savefig(self.filekey + '_cat_dist.png')

    def all_error_analysis(self, years, numeric_exc_binary, numeric, categorical):
        '''Takes (1) number of years, (2) a list of numeric column names excluding binary
        (3) a list of numeric column names including binary and (4) a list of categorical variables,
        Runs all analyses.'''
        self.create_err_analysis_df(2)
        self.divide_preds()
        self.plot_boxplots(numeric_exc_binary)
        self.mean_comparison(numeric)
        self.correlation(numeric)
        self.make_cat_breakdowns(categorical)
        self.plot_cat_breakdowns() 