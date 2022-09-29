"""
Module to perform basic churn prediction modelling on credit card data.

author: Markus Wehr
date: 2022-09-29
"""


import os
import shap
import joblib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


os.environ['QT_QPA_PLATFORM']='offscreen'


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''	
    df = pd.read_csv(pth)
    
    return df


def create_churn_target(df):
    '''
    Create churn variable from "Attrition_Flag" feature

    input:
        df: pandas dataframe
    
    output:
        df: pandas dataframe
    '''
    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == 'Existing Customer' else 1)

    return df


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # Histogramm for binary "Churn" variable
    plt.figure(figsize=(20,10)) 
    df['Churn'].hist()
    plt.savefig('./images/eda/churn_hist.png');

    # Histogramm for continuous "Customer_Age" variable
    plt.figure(figsize=(20,10)) 
    df['Customer_Age'].hist()
    plt.savefig('./images/eda/customer_age_hist.png');

    # Barchart for discrete "Marital_Status" variable
    plt.figure(figsize=(20,10)) 
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig('./images/eda/marital_status_bar.png');

    # KDE plot for "Total_Trans_Ct" variable (number of transactions per customer)
    plt.figure(figsize=(20,10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig('./images/eda/total_transact_ct_kde.png');

    # Correlation heatmap for all variables
    plt.figure(figsize=(20,10)) 
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig('./images/eda/correlation_heatmap.png');

    return None


def encoder_helper(df, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features

    output:
            df: pandas dataframe with new columns for proportion of churn
    '''
    # Iterate over all categorical columns
    for col in category_lst:
        col_lst = []
        # Groupby given column and get mean churn per category
        col_groups = df.groupby(col).mean()['Churn']

        # Iterate over column's rows
        for val in df[col]:
            # Append mean churn for each category of a given row
            col_lst.append(col_groups.loc[val])

        # Add newly created column to dataframe
        df[f'{col}_Churn'] = col_lst

    return df


def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    # Define target variable
    y = df['Churn']

    # Define independent features
    X = pd.DataFrame()
    keep_cols = [
            'Customer_Age', 'Dependent_count', 'Months_on_book',
            'Total_Relationship_Count', 'Months_Inactive_12_mon',
            'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
            'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
            'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
            'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn', 
            'Income_Category_Churn', 'Card_Category_Churn'
            ]
    X[keep_cols] = df[keep_cols]

    # Define train and test set for X and y respectively
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
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
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/rf_classification_report.png')

    # Plotting classification report for Logistic Regression and saving as image
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
    plt.axis('off')
    plt.savefig('./images/results/lr_classification_report.png');

    return None


def feature_importance_plot(model, X_data, output_pth):
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
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20,5))

    # Create plot title
    plt.title('Feature Importance')
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    
    # Save image
    plt.savefig(output_pth);

    return None


def train_models(X_train, X_test, y_train, y_test):
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
    param_grid = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth' : [4,5,100],
        'criterion' :['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    # Use models to generate predictions from test set
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Plot ROC curves for both models and save to ./images/results
    for data_split in [[X_train, y_train, 'train'], [X_test, y_test, 'test']]:
        lrc_plot = plot_roc_curve(lrc, data_split[0], data_split[1])
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, data_split[0], data_split[1], ax=ax, alpha=0.8)
        lrc_plot.plot(ax=ax, alpha=0.8)
        plt.savefig(f'./images/results/roc_curves_{data_split[2]}.png');

    # Save best models
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Create and save classification reports
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf
        )
    
    # Create feature importance plot and save in directory
    feature_importance_plot(
        cv_rfc, X_train, 
        output_pth='./images/results/feature_importance.png',
        )

    return None

if __name__ == '__main__':
    DATA_PATH = './data/bank_data.csv'
    CATEGORY_LST = [
    'Gender', 'Education_Level', 'Marital_Status',
    'Income_Category', 'Card_Category',
    ]
    
    bank_data = import_data(DATA_PATH)
    bank_data = create_churn_target(bank_data)
    perform_eda(bank_data)
    bank_data = encoder_helper(bank_data, CATEGORY_LST)
    X_train, X_test, y_train, y_test = perform_feature_engineering(bank_data)
    train_models(X_train, X_test, y_train, y_test)
