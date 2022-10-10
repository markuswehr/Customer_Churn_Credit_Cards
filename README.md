# Predict Customer Churn (ML Engineering DevOps)

Python package to predict customer churn (data: 
https://www.kaggle.com/datasets/sakshigoyal7/credit-card-customers/code). 
Class project for Udacity's ML Engineering DevOps nano-degree.

## Motivation

This project is part of Udacity's ML Engineering DevOps nano-degree. It is the first of four assignments. Using these data, I am putting clean code principles to practice.

## Features

With this library, one can perform data loading, EDA, feature engineering and modelling in one command. Thereby, wrapping together the formerly fragmented process of customer churn prediction.

## Code Example

After installing all required libraries, one can execute the code by simply running

```
python churn_library.py
```

in command line.

## Installation

Before running the library, all required dependencies must be installed. Run the following command to do so:

```
pip install -r requirements_py3.8.txt
```

## Tests

The library also contains unittests. To run tests:

```
pytest churn_script_logging_and_tests.py
```

To run tests AND save log files:

```
python churn_script_logging_and_tests.py
```

## Files and data description

Below gives an overview of the project structure and a short description of what it contains:

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
