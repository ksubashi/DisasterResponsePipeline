import sys
import pandas as pd
from sqlalchemy import create_engine
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import KNeighborsClassifier
from sklearn.externals import joblib
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','stopwords'])


def load_data(database_filepath):
    '''
    Args:
        database_filepath  : (relative) filepath of sqlite database
    Returns:
        X and Y as feature and target dataset
    '''
    #Connect to DB and access Messages table
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Messages', engine) 
    #X and Y definition
    X = df.message
    Y = df.iloc[:,4:]
    #Turn column names from Y as target names stored in an array
    target_names=Y.columns.tolist()
    return X,Y,target_names


def tokenize(text):
    '''
    Args:
        text : Message column which needs NLP preprocessing
    Returns:
        tokens cleaned
    '''
    #Lower , remove non laphanumeric characters, tokenize and remove stop words from text
    text=text.lower()
    text=re.sub(r"[^a-zA-Z0-9]", " ",text)
    words=word_tokenize(text)
    words=[w for w in words if w not in stopwords.words("english")]
    
    lemmatizer=WordNetLemmatizer()
    clean_tokens=[]
    #lemmatize text and append tokens in array
    for t in words:
        tokens=lemmatizer.lemmatize(t).strip()
        clean_tokens.append(tokens)
    return clean_tokens


def build_model():
    '''
    Args:
        None
    Returns:
        A pipeline object
    '''
    pipeline = Pipeline([
    ('count_vect',CountVectorizer(tokenizer=tokenize)),
    ('tfidf',TfidfTransformer()),
    ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    param_grid = {
    'n_estimators': [50,100],
    'max_depth': [None, 5]
    }

    # Instantiate GridSearchCV
    grid_search = GridSearchCV(pipeline, param_grid, cv=5)
    return grid_search


def evaluate_model(model, X_test, Y_test, category_names):
    '''Args:
        model          : Makes the predictions 
        X_test         : Features to be used for predictions
        Y_test         : Target Value
        target_names   : Name of Target Columns
    Returns:
        None
    '''
    #Predict values from X_test
    Y_pred=model.predict(X_test)
    #Initialize indx for index as -1 and in the for loop add +1 to start on index 0 for the first column
    indx=-1
    for column in category_names:
        indx+=1
        print(column)
        print("_"*50)
        #Print results of predictions made
        print(classification_report(Y_test[column],Y_pred[:,indx]))


def save_model(model, model_filepath):
    '''Args:
        model          : Makes the predictions 
        model_filepath : path where to keep the pickled model
    Returns:
        None
    '''
    joblib.dump(model,model_filepath,compress=3)


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
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()