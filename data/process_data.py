'''
This is ETL Pipeline is handling two datasets
1- messages.csv is contain the messages
2- categories.csv is contain the categories for each message

Frist: will load the datasets into dataframe and drop unusful columns and merge them
Second: will clean the dataframe by fix catefories and drop duplicates
Third: will save the new data into SQL database

'''
# import libraries
import pandas as pd
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    '''
    Will load messages and categories from CSV to dataframe and drop unusful columns
    Inputs : messages.csv and categories.csv
    Output : DataFrame merge both files contain only id, messages, and categories.
    '''
    # load messages dataset dataframe
    
    messages = pd.read_csv(messages_filepath)
    
    # load categories dataset dataframe
    categories = pd.read_csv(categories_filepath)
    
    # Drop original, genre from muessages
    
    #messages = messages.drop(['original', 'genre'], axis =1)
    
    # Merge the messages and categories datasets using the common id
    df = pd.merge(messages, categories, on='id')
    
    return df
    

def clean_data(df):
    '''
    Will use categories data and split into separate category columns and drop the duplicates 
    Inputs : DataFrame contain id, messages, and categories.
    Output : DataFrame merge both files contain only id, messages, and categories separated into category columns and droped the duplicates.
    '''
    # First row of the categories dataframe as a list
    categories_names_lst = df.categories.values[0].split(';')
    categories_names = []
    for col_name in categories_names_lst:
        categories_names.append(col_name[:-2])
    
    # Create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(';', expand =True)
    
    # Update columns names
    categories.columns = categories_names
    
    # Convert category values to just numbers and change them to numeric

    for column in categories:
        # Convert the Series to be of type string
        categories[column] = categories[column].astype(str)

        # Convert each value to be the last character of the string
        categories[column] = categories[column].str[-1:]

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

        # To fix values that equal 2
        categories[column] = categories[column].apply(lambda x:1 if x>1 else x)
    
    # Join the data and drop categories since it is no longer needed
    df = pd.concat([df,categories], axis =1)
    df = df.drop('categories', axis =1)
    
    # Drop the duplicates from the dataframe
    df = df.drop_duplicates()
    
    return df

def save_data(df, database_filename):
    '''
    Will save dataframe to sql database
    Inputs : dataframe, database name
    Output : there no output is only save the database in current path and you will can access to it by the database name
    '''
    
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_messages', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()