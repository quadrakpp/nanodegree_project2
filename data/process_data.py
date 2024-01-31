# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Loading datasets.
    Args:
        messages_filepath : messages dataset file path.
        categories_filepath : categories dataset file path.
    Returns:
        df : The merged one.
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on='id', how='inner')
    
    return df


def clean_data(df):
    """
    Cleaning the input dataset.
    Args:
        df : input dataset.
    Returns:
        df_final : The cleaned one.
    """
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0, :]
    category_colnames = [i.split('-')[0] for i in row]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].apply(lambda x: x.split('-')[-1])
        categories[column] = categories[column].astype(int)
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    df_final = df.drop_duplicates()
    
    return df_final


def save_data(df, database_filename):
    """
    Save the dataset to sqlite database.
    Args:
        df : dataset.
        database_filename : sqlite database filename.
    Returns:
        Nothing
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('InsertTableName', engine, index=False, if_exists='replace')


def main():
    '''
    Run Data Pipeline
    '''
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