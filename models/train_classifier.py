
'''
This is Machine learning Pipeline is dealing with SQL database and train a model
Frist will load the database and create messages and target dataframes
Second will use these two dataframes to train the model
Third: will save the model into a pickle model

'''

# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
import pickle
import sys


def load_data(database_filepath):
	'''
    Will load messages and categories from SQL database and create training and targets, with the name of targets list.
    Inputs : Database contain the messages and targets categories.
    Output : Two series for messages (X) and targets (Y) also list contain the names of targets (category_names).
    '''
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table('disaster_messages',engine)
    X = df['message']
    Y = df.drop(['id','message','original', 'genre'], axis=1)
    category_names = Y.columns.tolist()
    
    return X, Y, category_names


def tokenize(text):
	'''
    Will tokenize the messages and lemmatize them
    Inputs : Messages text.
    Output : List of words that explain the message.
    '''

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
 	'''
    Will create pipeline for build the model that contain transformation process and estimator
    Inputs : no need to input.
    Output : grid serach that contain pipeline transorm normal messages to format understood by NLP.
    '''

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))   
        ])

    parameters = {'clf__estimator__n_estimators': [50, 30],
              'clf__estimator__min_samples_split': [3, 2] 
    }

    cv = GridSearchCV(pipeline, param_grid= parameters, verbose=2)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
	'''
    Will test and report how the model perform in data unseen yet.
    Inputs : trained model, data unssen yet from messages (X) and targets (Y), and list contain the names of targets (category_names) 
    Output : no returned values but there a report will show how the model perform.
    '''
    y_pred = model.predict(X_test)
    print(classification_report(y_pred, Y_test, target_names = category_names))


def save_model(model, model_filepath):
	'''
    Will save the model into a pickle file so we can use in predcation.
    Inputs : the trained model and the name of pickle file what will warp the model. 
    Output : no returned values but will save the model in the application folder.
    '''
    pickle.dump(model, open(model_filepath, 'wb'))


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