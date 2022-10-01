"""
Module to perform basic churn prediction modelling on credit card data.

author: Markus Wehr
date: 2022-09-29
"""


import constant_vars as cons
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import train_test_split

import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


os.environ['QT_QPA_PLATFORM'] = 'offscreen'


class CreditCardChurn():
    '''
    Class to process credit card data for predicting customer churn
    '''

    def __init__(self, pth) -> None:
        '''
        Initiating CreditCardChurn class

        input:
            pth: String path to bank data
        output:
            self.path: String variable with path to csv file
        '''
        self.path = pth
        self.bank_data = pd.DataFrame()
        self.X_all = pd.DataFrame()
        self.X_train = pd.DataFrame()
        self.X_test = pd.DataFrame()
        self.y_all = pd.Series()
        self.y_train = pd.Series()
        self.y_test = pd.Series()

    def import_data(self) -> None:
        '''
        returns dataframe for the csv found at pth

        input:
            self.path: String with path to csv file
        output:
            self.bank_data: Pandas dataframe with bank data
        '''
        self.bank_data = pd.read_csv(self.path)

    def create_churn_target(self) -> None:
        '''
        Create churn variable from "Attrition_Flag" feature

        input:
            self.bank_data: Pandas dataframe after importing from csv
        output:
            self.bank_data: Pandas dataframe with churn variable created
        '''
        self.bank_data['Churn'] = self.bank_data['Attrition_Flag'].apply(
            lambda val: 0 if val == 'Existing Customer' else 1)

    def perform_eda(self) -> None:
        '''
        perform eda on df and save figures to images folder
        input:
            self.bank_data: Pandas dataframe (incl. churn variable)
        output:
            None
        '''
        # Histogramm for binary "Churn" variable
        plt.figure(figsize=(20, 10))
        self.bank_data['Churn'].hist()
        plt.savefig('./images/eda/churn_hist.png')

        # Histogramm for continuous "Customer_Age" variable
        plt.figure(figsize=(20, 10))
        self.bank_data['Customer_Age'].hist()
        plt.savefig('./images/eda/customer_age_hist.png')

        # Barchart for discrete "Marital_Status" variable
        plt.figure(figsize=(20, 10))
        self.bank_data.Marital_Status.value_counts(
            'normalize').plot(kind='bar')
        plt.savefig('./images/eda/marital_status_bar.png')

        # KDE plot for "Total_Trans_Ct" variable (number of transactions per
        # customer)
        plt.figure(figsize=(20, 10))
        sns.histplot(
            self.bank_data['Total_Trans_Ct'],
            stat='density',
            kde=True)
        plt.savefig('./images/eda/total_transact_ct_kde.png')

        # Correlation heatmap for all variables
        plt.figure(figsize=(20, 10))
        sns.heatmap(
            self.bank_data.corr(),
            annot=False,
            cmap='Dark2_r',
            linewidths=2)
        plt.savefig('./images/eda/correlation_heatmap.png')

    def encoder_helper(self, category_lst):
        '''
        helper function to turn each categorical column into a new column with
        propotion of churn for each category - associated with cell 15 from the notebook

        input:
            category_lst: list of columns that contain categorical features

        output:
            self.bank_data: Pandas dataframe with encoded categorical cols
        '''
        # Iterate over all categorical columns
        for col in category_lst:
            col_lst = []
            # Groupby given column and get mean churn per category
            col_groups = self.bank_data.groupby(col).mean()['Churn']

            # Iterate over column's rows
            for val in self.bank_data[col]:
                # Append mean churn for each category of a given row
                col_lst.append(col_groups.loc[val])

            # Add newly created column to dataframe
            self.bank_data[f'{col}_Churn'] = col_lst

    def perform_feature_engineering(self, category_lst) -> None:
        '''
        Creates train and test sets for further modelling

        input:
            category_lst: list of columns that contain categorical features
        output:
            self.X_train: X training data
            self.X_test: X testing data
            self.y_train: y training data
            self.y_test: y testing data
        '''
        # Proportion of churn per category variables
        self.encoder_helper(category_lst=category_lst)

        # Define target variable
        self.y_all = self.bank_data['Churn']

        # Define independent features
        self.X_all[cons.KEEP_COLS] = self.bank_data[cons.KEEP_COLS]

        # Define train and test set for X and y respectively
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_all, self.y_all, test_size=0.3, random_state=42)

    def classification_report_image(self,
                                    y_train_preds_lr,
                                    y_train_preds_rf,
                                    y_test_preds_lr,
                                    y_test_preds_rf):
        '''
        produces classification report for training and testing results and stores report as image
        in images folder
        input:
                y_train: training response values
                y_test:  test response values
                y_train_preds_lr: training predictions from logistic regression
                y_train_preds_rf: training predictions from random forest
                y_test_preds_lr: test predictions from logistic regression
                y_test_preds_rf: test predictions from random forest

        output:
                None
        '''
        # Plotting classification report for Random Forest and saving as image
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Random Forest Train'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(self.y_test, y_test_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Random Forest Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(self.y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig('./images/results/rf_classification_report.png')

        # Plotting classification report for Logistic Regression and saving as
        # image
        plt.rc('figure', figsize=(5, 5))
        plt.text(0.01, 1.25, str('Logistic Regression Train'),
                 {'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.05, str(classification_report(self.y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.text(0.01, 0.6, str('Logistic Regression Test'), {
            'fontsize': 10}, fontproperties='monospace')
        plt.text(0.01, 0.7, str(classification_report(self.y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')  # approach improved by OP -> monospace!
        plt.axis('off')
        plt.savefig('./images/results/lr_classification_report.png')

    def feature_importance_plot(self, model, output_pth):
        '''
        creates and stores the feature importances in pth
        input:
                model: model object containing feature_importances_
                X_data: pandas dataframe of X values
                output_pth: path to store the figure

        output:
                None
        '''
        # Calculate feature importances
        importances = model.best_estimator_.feature_importances_
        # Sort feature importances in descending order
        indices = np.argsort(importances)[::-1]

        # Rearrange feature names so they match the sorted feature importances
        names = [self.X_all.columns[i] for i in indices]

        # Create plot
        plt.figure(figsize=(20, 5))

        # Create plot title
        plt.title('Feature Importance')
        plt.ylabel('Importance')

        # Add bars
        plt.bar(range(self.X_all.shape[1]), importances[indices])

        # Add feature names as x-axis labels
        plt.xticks(range(self.X_all.shape[1]), names, rotation=90)

        # Save image
        plt.savefig(output_pth)

    def train_models(self):
        '''
        train, store model results: images + scores, and store models
        input:
                X_train: X training data
                X_test: X testing data
                y_train: y training data
                y_test: y testing data
        output:
                None
        '''
        # Perform grid search on both models and fit to train set
        rfc = RandomForestClassifier(random_state=42)
        lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
        cv_rfc = GridSearchCV(estimator=rfc, param_grid=cons.PARAM_GRID, cv=5)
        cv_rfc.fit(self.X_train, self.y_train)
        lrc.fit(self.X_train, self.y_train)

        # Use models to generate predictions from test set
        y_train_preds_rf = cv_rfc.best_estimator_.predict(self.X_train)
        y_test_preds_rf = cv_rfc.best_estimator_.predict(self.X_test)
        y_train_preds_lr = lrc.predict(self.X_train)
        y_test_preds_lr = lrc.predict(self.X_test)

        # Plot ROC curves for both models and save to ./images/results
        for data_split in [[self.X_train, self.y_train, 'train'], [
                self.X_test, self.y_test, 'test']]:
            lrc_plot = plot_roc_curve(lrc, data_split[0], data_split[1])
            plt.figure(figsize=(15, 8))
            axis = plt.gca()
            plot_roc_curve(
                cv_rfc.best_estimator_,
                data_split[0],
                data_split[1],
                ax=axis,
                alpha=0.8)
            lrc_plot.plot(ax=axis, alpha=0.8)
            plt.savefig(f'./images/results/roc_curves_{data_split[2]}.png')
            plt.gca().cla()

        # Save best models
        joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
        joblib.dump(lrc, './models/logistic_model.pkl')

        # Create and save classification reports
        self.classification_report_image(
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf
        )

        # Create feature importance plot and save in directory
        self.feature_importance_plot(
            cv_rfc,
            output_pth='./images/results/feature_importance.png',
        )

    def main(self, category_lst):
        '''
        main() function to execute if __name__ == "__main__" or to call when importing the module
        to run the whole workflow

        input:
            pth: a path to the csv
            categroy_lst: list of columns that contain categorical features
        output:
            None
        '''
        self.import_data()
        self.create_churn_target()
        self.perform_eda()
        self.perform_feature_engineering(category_lst=category_lst)
        self.train_models()


if __name__ == '__main__':
    credit_churn = CreditCardChurn(cons.DATA_PATH)
    credit_churn.main(cons.CATEGORY_LST)
