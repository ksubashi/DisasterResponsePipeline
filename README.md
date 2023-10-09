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

1. Run the following commands in the project's root directory to set up your database and model.
    *    To run ETL Pipeline, run below command to clean data and saved in into Database:
        'python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db'
    *    To run ML pipeline that trains classifier and saves on shell
        'python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl'
2. Run the following commands  to run the web app.
    *    'cd app'
    *    'python run.py'

## Project Motivation<a name="motivation"></a>

Dividing a fake help request from a real request for help is a real life problem and finding the solution would be so beneficial. In this project I am using these data to solve this real problem
by using machine learning algorithms and ETL steps for everything. The data are provided from FigureEight and going through them might be interesting


## File Descriptions <a name="files"></a>

1. ETL Pipeline Preparation.ipynb and ML Pipeline Preparation.ipynb are both jupyter notebook files which are used for testing purposes , and stores all steps used in .py files.
2. app folder:
   * 'run.py'- Flask file that runs app
   * templates:
         * master.html - main page of web app
         * go.html - classification result page of web app
3. data folder:
   * DisasterResponseDB.db - Database where cleaned dataframe is stored
   * diaster_categories.csv and disaster_messages.csv - data to be processed and analyzed
   * process_data.py - python file where all functions are stored to ETL the files
4. models folder:
   * Classifier.pkl - saved model
   * train_classifier.py - python file where all machine learning algorithms are stored
   

## Results<a name="results"></a>
The model performed pretty well after optimization techniques.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to FigureEight for the data.  You can find the Licensing for the data and other descriptive information at the FigureEight link available [here](https://appen.com/).  Otherwise, feel free to use the code here as you would like! 
