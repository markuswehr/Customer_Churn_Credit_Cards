'''
Module to test functions in churn_library.py using pytest

author: Markus Wehr
date: 2022-09-29
'''


import os
import logging

from churn_library import CreditCardChurn
import constant_vars as cons

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


class TestChurnLibrary():
    '''
    Class to test different functions of churn_library module
    '''

    def test_import(self):
        '''
        test data import - this example is completed for you to assist with the other test functions
        '''
        # Test wether file is found in given path
        try:
            churn_library_class = CreditCardChurn(cons.DATA_PATH)
            churn_library_class.import_data()
            logging.info("SUCCESS: Testing import_data - File found")
        except FileNotFoundError as err:
            logging.error("ERROR: Testing import_data - The file wasn't found")
            raise err

        # Test wether dataframe has columns and rows
        try:
            assert churn_library_class.bank_data.shape[0] > 0
            assert churn_library_class.bank_data.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "ERROR: Testing import_data - The file doesn't appear to have rows and columns")
            raise err

    def test_eda(self, test_class):
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
                assert col in test_class.bank_data.columns, \
                    f'ERROR: Testing perform_eda - {col} column not in dataframe'
            logging.info(
                'SUCCESS: Testing perform_eda - Found all required columns in dataframe')
        except AssertionError as err:
            logging.error(err)

        # Assert if all images are saved to ./image directory
        try:
            #churn_library_class = CreditCardChurn(cons.DATA_PATH)
            test_class.create_churn_target()
            test_class.perform_eda()
            image_names = [
                'churn_hist', 'customer_age_hist', 'marital_status_bar',
                'total_transact_ct_kde', 'correlation_heatmap',
            ]
            for image in image_names:
                assert os.path.isfile(
                    f'./images/eda/{image}.png') is True, \
                    f'ERROR: Testing perform_eda - \
                            "{image}.png" was not saved in "./image/eda" directory'
            logging.info(
                'SUCCESS: Testing perform_eda - Saved all images to "./images/eda" directory')
        except AssertionError as err:
            logging.error(err)

    def test_encoder_helper(self, test_class):
        '''
        test encoder helper
        '''
        try:
            # Check if all categorical columns are actually in the data
            for col in cons.CATEGORY_LST:
                assert col in test_class.bank_data.columns, \
                    f'ERROR: Testing encoder_helper - {col} column not in dataframe'

            # Assert that as many columns were newly created as there are in the
            # categorical columns list
            test_class.create_churn_target()
            test_class.encoder_helper(cons.CATEGORY_LST)
            new_cols = [
                col for col in test_class.bank_data.columns if col.endswith('_Churn')]
            assert len(new_cols) == len(
                cons.CATEGORY_LST), \
                'ERROR: Testing encoder_helper - Did not transform all categorical columns'

            logging.info(
                'SUCCESS: Testing econder_helper - Added mean churn rate per categorical column')
        except AssertionError as err:
            logging.error(err)

    def test_perform_feature_engineering(self, test_class):
        '''
        test perform_feature_engineering
        '''
        # Run function to see if there are any key errors
        try:
            test_class.create_churn_target()
            test_class.perform_feature_engineering(cons.CATEGORY_LST)
            logging.info(
                'SUCCESS: Testing perform_feature_engineering - Created train and test set')
        except KeyError:
            logging.error(
                'ERROR: Testing perfom_feature_engineering - \
                    Could not find all columns specified by "keep_cols"')

        try:
            test_class.create_churn_target()
            test_class.encoder_helper(cons.CATEGORY_LST)
            test_class.perform_feature_engineering(cons.CATEGORY_LST)

            # Assert if the train and test features are of same lenghts as the
            # train and test targets
            assert len(test_class.X_train) == len(
                test_class.y_train), \
                'ERROR: Testing perform_feature_engineering - \
                        X and y train sets are not of the same lenght'
            assert len(test_class.X_test) == len(
                test_class.y_test), \
                'ERROR: Testing perform_feature_engineering - \
                        X and y test sets are not of the same lenght'

        except AssertionError as err:
            logging.error(err)

    def test_train_models(self, test_class):
        '''
        test train_models
        '''
        try:
            test_class.create_churn_target()
            test_class.encoder_helper(cons.CATEGORY_LST)
            test_class.perform_feature_engineering(cons.CATEGORY_LST)
            test_class.train_models()

            # Check wether all results were saved in directory
            image_list = [
                'roc_curves_train', 'roc_curves_test',
                'feature_importance', 'rf_classification_report',
                'lr_classification_report',
            ]
            for image in image_list:
                assert os.path.isfile(
                    f'./images/results/{image}.png') is True, \
                    f'ERROR: Testing train_models - \
                            "{image}.png" was not saved in "./image/results" directory'

            # Check wether both models were saved in directory
            model_list = [
                'rfc_model', 'logistic_model'
            ]
            for model in model_list:
                assert os.path.isfile(
                    f'./models/{model}.pkl') is True, \
                    f'ERROR: Testing train_models - \
                            "{model}.pkl" was not saved in "./models" directory'

            logging.info(
                'SUCCESS: Testing train_models - \
                    Models trained and saved in "./models"; Results saved in "./images/results"')
        except AssertionError as err:
            logging.error(err)


if __name__ == "__main__":
    # Create CreditCardChurn class and import data (replaces fixture when run
    # from cmd)
    credit_card_churn_class = CreditCardChurn(cons.DATA_PATH)
    credit_card_churn_class.import_data()

    # Create TestChrunLibrary class and run different test functions
    churn_library_tests = TestChurnLibrary()
    churn_library_tests.test_import()
    churn_library_tests.test_eda(credit_card_churn_class)
    churn_library_tests.test_encoder_helper(credit_card_churn_class)
    churn_library_tests.test_perform_feature_engineering(
        credit_card_churn_class)
    churn_library_tests.test_train_models(credit_card_churn_class)
