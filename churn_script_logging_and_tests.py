'''
Module to unit-test functions in churn_library.py

author: Markus Wehr
date: 2022-09-29
'''

from cgi import test
import os
import logging
import pytest
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

DATA_PATH = './data/bank_data.csv'
CATEGORY_LST = [
    'Gender', 'Education_Level', 'Marital_Status',
    'Income_Category', 'Card_Category',
    ]


# Create test_data fixture to resuse for multiple tests
@pytest.fixture
def test_data():
    df = cls.import_data(DATA_PATH)

    return df


def test_import():
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
    # Test wether file is found in given path
	try:
		df = cls.import_data(DATA_PATH)
		logging.info("SUCCESS: Testing import_data - File found")
	except FileNotFoundError as err:
		logging.error("ERROR: Testing import_data - The file wasn't found")
		raise err

    # Test wether dataframe has columns and rows
	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("ERROR: Testing import_data - The file doesn't appear to have rows and columns")
		raise err


def test_eda(test_data):
    '''
    test perform eda function
    '''
    # Test if necessary columns are present in df
    try:
        required_columns = [
            'Attrition_Flag', 'Customer_Age',
            'Marital_Status', 'Total_Trans_Ct',
            ]
        for col in required_columns:
            assert col in test_data.columns, f'ERROR: Testing perform_eda - {col} column not in dataframe'
        logging.info('SUCCESS: Testing perform_eda - Found all required columns in dataframe')
    except AssertionError as err:
        logging.error(err)

    # Assert if all images are saved to ./image directory
    try:
        test_data_with_churn = cls.create_churn_target(test_data)
        cls.perform_eda(test_data_with_churn)
        image_names = [
            'churn_hist', 'customer_age_hist', 'marital_status_bar',
            'total_transact_ct_kde', 'correlation_heatmap',
            ]
        for image in image_names:
            assert os.path.isfile(f'./images/eda/{image}.png') is True, f'ERROR: Testing perform_eda - "{image}.png" was not saved in "./image/eda" directory'
        logging.info('SUCCESS: Testing perform_eda - Saved all images to "./images/eda" directory')
    except AssertionError as err:
        logging.error(err)


def test_encoder_helper(test_data):
    '''
    test encoder helper
    '''
    try:
        # Check if all categorical columns are actually in the data
        for col in CATEGORY_LST:
            assert col in test_data.columns, f'ERROR: Testing encoder_helper - {col} column not in dataframe'
        
        # Assert if churn rate column is of type int
        encoded_test_data = cls.create_churn_target(test_data)
        assert isinstance(encoded_test_data.Churn.dtype, int), f'ERROR: Testing encoder_helper - "Churn" columns is not of type int'
        
        # Assert that as many columns were newly created as there are in the categorical columns list
        encoded_test_data = cls.encoder_helper(encoded_test_data, CATEGORY_LST)
        new_cols = [col for col in encoded_test_data if col.endswith('_Churn')]
        assert len(new_cols) == len(CATEGORY_LST), f'ERROR: Testing encoder_helper - Did not transform all categorical columns'
        
        # Assert that the newly created churn rate by group columns are of type float
        for col in new_cols:
            assert isinstance(encoded_test_data[col].dtype, float), f'ERROR: Testing encoder_helper - {col} column is not of type float'
        
        logging.info('SUCCESS: Testing econder_helper - Added mean churn rate per categorical column')
    except AssertionError as err:
        logging.error(err)


def test_perform_feature_engineering(test_data):
    '''
    test perform_feature_engineering
    '''
    # Run function to see if there are any key errors
    try:
        encoded_test_data = cls.create_churn_target(test_data)
        encoded_test_data = cls.encoder_helper(encoded_test_data, CATEGORY_LST)
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(encoded_test_data)
        logging.info('SUCCESS: Testing perform_feature_engineering - Created train and test set')
    except KeyError:
        logging.error('ERROR: Testing perfom_feature_engineering - Could not find all columns specified by "keep_cols"')

    try:
        encoded_test_data = cls.create_churn_target(test_data)
        encoded_test_data = cls.encoder_helper(encoded_test_data, CATEGORY_LST)
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(encoded_test_data)

        # Assert if the train and test features are of same lenghts as the train and test targets
        assert len(X_train) == len(y_train), f'ERROR: Testing perform_feature_engineering - X and y train sets are not of the same lenght'
        assert len(X_test) == len(y_test), f'ERROR: Testing perform_feature_engineering - X and y test sets are not of the same lenght'
        
        # Assert if train and test targets are of type int
        assert isinstance(y_train.dtype, int), f'ERROR: Testing perform_feature_engineering - y_train is not if type int'
        assert isinstance(y_test.dtype, int), f'ERROR: Testing perform_feature_engineering - y_test is not of type int'
    except AssertionError as err:
        logging.error(err)


def test_train_models(test_data):
    '''
    test train_models
    '''
    try:
        encoded_test_data = cls.create_churn_target(test_data)
        encoded_test_data = cls.encoder_helper(encoded_test_data, CATEGORY_LST)
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(encoded_test_data)
        cls.train_models(X_train, X_test, y_train, y_test)
        
        # Check wether all results were saved in directory
        image_list = [
            'roc_curves_train', 'roc_curves_test',
            'feature_importance', 'rf_classification_report', 
            'lr_classification_report',
            ]
        for image in image_list:
            assert os.path.isfile(f'./images/results/{image}.png') is True, f'ERROR: Testing train_models - "{image}.png" was not saved in "./image/results" directory'
        
        # Check wether both models were saved in directory
        model_list = [
                'rfc_model', 'logistic_model'
            ]
        for model in model_list:
            assert os.path.isfile(f'./models/{model}.pkl') is True, f'ERROR: Testing train_models - "{model}.pkl" was not saved in "./models" directory'
        
        logging.info('SUCCESS: Testing train_models - Models trained and saved in "./models"; Results saved in "./images/results"')
    except AssertionError as err:
        logging.error(err)


if __name__ == "__main__":
    test_import()
    test_eda()
    test_encoder_helper()
    test_perform_feature_engineering()
    test_train_models()
