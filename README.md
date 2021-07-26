# abtest-mlops

**Table of content**

- [abtest-mlops](#abtest-mlops)
  - [Overview](#overview)
  - [Scenario](#scenario)
  - [Objective](#objective)
  - [Approach](#approach)
  - [Project Structure](#project-structure)
    - [data:](#data)
    - [features:](#features)
    - [models:](#models)
    - [notebooks:](#notebooks)
    - [scripts](#scripts)
    - [tests:](#tests)
    - [root folder](#root-folder)
  - [Installation guide](#installation-guide)

## Overview
This repository is used for week 2 challenge of 10Academy. The instructions for this project can be found in the challenge document.

## Scenario
An advertising company called SmartAd is running an online ad for a client with the intention of
increasing brand awareness for the brand ‘Lux’. SmartAd provides an additional service called
Brand Impact Optimiser (BIO), a lightweight questionnaire, served with every campaign to
determine the impact of their creative ad design. SmartAd ran this campaign from 3-10 July
2020.

## Objective
The main objective of this project is to test if this advertising campaign resulted in a significant
lift in brand awareness. This means we are testing to see if the creative ad designed by SmartAd has performed significantly better than the dummy ad.

## Approach
The following three different types of A/B testing were used in the analysis.
1. Classic A/B testing
2. Sequential A/B testing
3. A/B testing with Machine Learning

For the machine learning part, I have used three machine learning algorithms
1. Logistic Regression
2. Decision Tree
3. XGBoost

## Project Structure
The repository has a number of files including python scripts, jupyter notebooks, pdfs and text files. Here is their structure with a brief explanation.

### data:
- the folder where the dataset csv file is stored

### features:
- the folder where the csv files with the features and targets sepated are stored

### models:
- the folder where models' pickle files are stored

### notebooks:
- `EDA.ipynb`: a jupyter notebook for exploratory data analysis
- `Classical A/B Testing.ipynb`: a jupyter notebook for A/B testing using Z-test
- `Sequential A/B Testing.ipynb`: a jupyter notebook for A/B testing using conditional SPRT
- `Machine Learning A/B Testing.ipynb`: a jupyter notebook for A/B testing using machine learning algorithms

### scripts
- `csv_helper.py`: a python script for handling reading and writing of csv files
- `df_cleaner.py`: a python script for cleaning pandas dataframes
- `df_selector.py`: a python script for selecting data from a pandas dataframe
- `df_outlier_handler.py`: a python script for cleaning outliers in  a pandas dataframe
- `config.py`: a python script for configuring path and other variables
- `stats.py`: a python script for computing statistics
- `stats.py`: a python script for computing statistics
- `plot.py`: a python script for plotting graphs in hypothesis testing
- `create_features.py`: a python script for splitting the training and testing datasets into features and targets datasets
- `train_model.py`: a python script for training a model using 5-fold cross validation and returns the best model
- `train_logistic.py`: a python script for training logistic regression
- `train_decision_tree.py`: a python script for training decision tree classifier
- `train_xgboost.py`: a python script for training XGBoost

### tests:
- the folder containing unit tests for components in the scripts

### root folder
- `10 Academy Batch 4 - Week 2 Challenge.pdf`: the challenge document
- `requirements.txt`: a text file lsiting the projet's dependancies
- `travis.yml`: a configuration file for Travis CI
- `setup.py`: a configuration file for installing the scripts as a package
- `README.md`: Markdown text with a brief explanation of the project and the repository structure.

## Installation guide
```
git clone https://github.com/zelalemgetahun9374/abtest-mlops
cd abtest-mlops
pip install -r requirements.txt
```