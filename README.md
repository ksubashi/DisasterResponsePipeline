# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#licensing)


## Installation <a name="installation"></a>

There should be no necessary libraries to run the code here beyond the Anaconda distribution of Python.  The code should run with no issues using Python versions 3.*.
About the data used , which are disaster_message.csv and disaster_categories.csv provided by FigureEight can be also found on data folder uploaded.

1.Run the following commands in the project's root directory to set up your database and model.
    To run ETL Pipeline, run below command to clean data and saved in into Database:
        'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
    To run ML pipeline that trains classifier and saves on shell
        'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
2. Run the following commands  to run the web app.
    'cd app'
    'python run.py'

## Project Motivation<a name="motivation"></a>

Dividing a fake help request from a real request for help is a real life problem and finding the solution would be so beneficial. In this project I am using these data to solve this real problem
by using machine learning algorithms and ETL steps for everything. The data are provided from FigureEight and going through them might be interesting


## File Descriptions <a name="files"></a>

ETL Pipeline Preparation.ipynb and ML Pipeline Preparation.ipynb are both jupyter notebook files which are used for testing purposes , and stores all steps used in .py files.

## Results<a name="results"></a>
The main findings of the code can be found at the post available [here](https://medium.com/@subashiklajbi/mastering-seattles-airbnb-pricing-and-tourist-demand-8ac714ddac6f).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Airbnb for the data.  You can find the Licensing for the data and other descriptive information at the Kaggle link available [here](https://www.kaggle.com/datasets/airbnb/seattle).  Otherwise, feel free to use the code here as you would like! 
