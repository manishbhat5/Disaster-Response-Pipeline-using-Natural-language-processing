# Disaster Response Pipeline Project

### Overview
This project contains a web app that asks for a message from a potential user who is in danger during a disaster and the app categorizes that message into a particular category such as aid related, weather related, fire or many more.

### Description of files
App: Contains javascript files and `app.py` file  which implements Flask & Plotly to create the web app
Data: Contains two CSV files `disaster_messages.csv` - Contains all the past messages & 'disaster_categories.csv' - contains the categories of the disaster messages
Model: Contains the Machine Learning Pipeline python script file to perform all the training and testing of the data

### List of Python libraries used
`nltk`
`sklearn`
`numpy`
`pandas`
`sys`
`sqlalchemy`
`re`
`pickle`
`json`
`flask`

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:8080/ 

### Snapshots
<p align="center">
  <img src="./img/example.png" alt="Size Limit CLI" width="738">
</p>
