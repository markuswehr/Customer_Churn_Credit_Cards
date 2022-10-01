'''
Module that defines fixtures for testing script

author: Markus Wehr
date: 2022-09-29
'''

import pytest

from churn_library import CreditCardChurn
import constant_vars as cons


# Create test_data fixture to resuse for multiple tests
@pytest.fixture
def test_class():
    '''
    Pytest fixture to avoid repeated importing of bank_data
    '''
    churn_library_class = CreditCardChurn(cons.DATA_PATH)
    churn_library_class.import_data()

    return churn_library_class
