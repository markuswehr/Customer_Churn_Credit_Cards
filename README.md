# Predict Customer Churn

## Project Description
Python package to predict customer churn (data: 
https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code). 
Class project for Udacity's ML Engineering DevOps nano-degree.

## Files and data description

```
Customer_Churn_Credit_Cards
├─ .gitignore --> Gitignore file
├─ Guide.ipynb --> Notebook with guidelines
├─ LICENSE --> MIT license
├─ README.md --> README including instructions and project description
├─ churn_library.py --> Main script to predict customer churn
├─ churn_notebook.ipynb --> Exploratory notebook
├─ churn_script_logging_and_tests.py --> Unittests
├─ conftest.py --> Pytest fixture for unittests
├─ constant_vars.py --> Constant variables to be used in churn_library.py
├─ data --> Csv file
├─ images --> Results and eda images
│  ├─ eda
│  └─ results
├─ logs --> Testing logs
├─ models --> Models saved as pkl files
├─ pytest.ini --> Pytest specs
└─ requirements_py3.8.txt --> Installation requirements

```

## Running Files

To install all required packages:

```
pip install -r requirements_py3.8.txt
```

To run churn analysis in chunr_library.py from cmd:

```
python churn_library.py
```

To run tests:

```
pytest churn_script_logging_and_tests.py
```

To run tests AND save log files:

```
python churn_script_logging_and_tests.py
```
