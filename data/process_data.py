import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Args:
        messages_filepath  : (relative) filepath of messages.csv
        categories_filepath: (relative) filepath of categories.csv

    Returns:        
        Returned to 'df'
    '''
    #Read messages csv
    messages=pd.read_csv(messages_filepath)
    #Read categories csv
    categories=pd.read_csv(categories_filepath)
    #Merge two datasets on id
    df=pd.merge(messages,categories,on='id')
    return df


def clean_data(df):
    '''
    Args:
        df: Pandas dataframe

    Returns:
        Returned to Cleaned category dataset that call 'df'
    '''    
    #Split categories column on ';' to make all categories as individual binary columns
    categories=df['categories'].str.split(";",expand=True)
    #Retrieve only first row
    row=categories.iloc[0]
    #From the first row access only category names without the value
    category_columns=row.apply(lambda x:x[:-2])
    #Use these category names as column names for all categories splitted before
    categories.columns=category_columns
    #Using for loop to access each one of the categories to get only numerical value(0,1) and cast to integer
    for column in categories.columns:
        categories[column] = categories[column].apply(lambda x:x[-1])
        categories[column] = categories[column].astype('int')
    #Drop original categories column from df
    df=df.drop(['categories'],axis=1)
    #Concat DF with categories dataframe created above
    df=pd.concat([df,categories],axis=1)
    #Drop all duplicates from dataframe
    df=df.drop_duplicates()
    
    return df
    


def save_data(df, database_filename):
    '''
    Args:
        df               : Pandas dataframe
        database_filename: (relative) filepath of sqlite database

    Returns:
        None
    '''
    #Create database and load table into DB
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Messages', engine, index=False)  


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