# import libraries
import sys
import re
import nltk
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
import pickle

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    """
    Load data from our sqlite database.
    Args:
        database_filepath : sqlite database filepath.
    Returns:
        X : dataframe for the 'message' feature.
        Y : dataframe for the 'target'.
        Category: list of feature names
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table('InsertTableName', con=engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    return X, Y, Y.columns

def tokenize(text):
    """
    text NLP processing (tokenizing, lemmatization).
    Args:
        text : incoming sentence.
    Returns:
        list of tokenized words.
    """
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    lemmed = [WordNetLemmatizer().lemmatize(w, pos='v') for w in lemmed]
    return lemmed 

def build_model():
    """
    building a pipeline and model.
    Args:
        Nothing
    Returns:
        ML model.
    """
    # pipeline definition
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(MultinomialNB()))
    ])
    
    # hyperparameters definition for tuning
    parameters = {
        'vect__ngram_range': [(1, 1), (1, 2), (1, 3)],
        'tfidf__use_idf': [True, False],
        'clf__estimator__alpha': [0.1, 0.5, 1.0]
    }
    
    ml_model = GridSearchCV(pipeline, parameters, verbose=1, cv=3)
    
    return ml_model
       
def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate the model and dispaly the clissification report.
    Args:
        model : ML model.
        X_test : Test dataset.
        Y_test : Test labeled data.
        category_names : category names.
    Returns:
        Nothing
    """
    Y_pred = model.predict(X_test)
    class_report = classification_report(Y_test, Y_pred, target_names=category_names)
    print(class_report)

def save_model(model, model_filepath):
    """
    Save the trained ML model to a pickle file.
    Args:
        model : ML model
        model_filepath : Destination path of the pickle file.
    Returns:
        Nothing
    """
    try:
        with open(model_filepath, 'wb') as file:  
            pickle.dump(model, file)
    except Exception as err:
        print(f"Somethimg wrong happened while saving the pickle file l: {str(e)}")


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()