# Disaster Response Pipeline Project

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## Project Motivation<a name="motivation"></a>

This project is about analyzing disaster data from Figure Eight and build a model for an API that classifies disaster messages. The project is combine these three (ETL Pipeline, Machine learning Pipeline, and web application) that use a trained model and classify any messages.

This project is designed using a real messages that were sent during disaster events, so this model can categorize these events to send the messages to an appropriate disaster relief agency.


## File Descriptions <a name="files"></a>

1. process_data.py : ETL Pipeline In a Python script (Load datasets, Merge datasets, Clean the data, and Store the data in a SQLite database).
2. train_classifier.py : Machine Learning Pipeline In a Python script  (Load data from SQLite database, Splits data into training and test sets, Builds a text processing and machine learning pipeline, Trains and tunes a model using GridSearchCV, Outputs results on the test set, and Exports the final model as a pickle file).
3. Run.py : Flask Web App (display visualization from the datasets, the app accept messages from users and returns classification results for 36 categories of disaster events).

## Licensing, Authors, Acknowledgements<a name="licensing"></a>
Must give credit to Udacity courses for build this challange and share the codes. Free to use the code here as you would like!
