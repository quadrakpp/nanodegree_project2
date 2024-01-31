# Description :
The aim of this project is to put in place a web site helping to categorize the need of people during the disaster crisis (Udacity Datascientis nanodegree program)

# Installation :
- We have to install Anaconda and python 3.10.9 in order to ensure that basics packages are installed like numpy, pandas, etc.
- Connect to Python prompt and with the help of 'pip install' install :
  
  **flask, pickle, plotly, nltk, SQLAlchemy**

# How to run the script :

- Connect to Python prompt and follow below steps :
  1. To launch the ETL pipeline :
     
     python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/disaster.db
     
  2. To launch the ML pipeline and create the model :
     
     python models/train_classifier.py data/disaster.db models/classifier.pkl
     (This operation can take several hours depending on your computer specs. It took me 4hours with 16GB of memory, 2.60GHz (8 CPUs))
     
  3. To launch the server web, move to **app** folder first and run :
     
     python run.py
     
     Once you get confirmation that the Server Flask is running the open your browser and go to http://127.0.0.1:3001

# Files descriptions :
  
1. Folder **app** :
    - run.py : The server launcher
    - templates : folder containing the web pages that the Flask server will load.
      
2. Folder **data** :
    - process_data.py : Python codes doing th ETL part.
    - disaster_messages.csv : Dataset of messages
    - disaster_categories.csv : Dataset of categories.
   
3. Folder **models** :
    - train_classifier.py : Python codes generating the model.
    - classifier.pkl : ML model generated as a pickle file

4. Folder **screenshots** : Contain some app frontend pages
