import yaml
from pathlib import Path

import pandas as pd

def load_document(path):
    """
    loads a directory and combines all parquet files into a single dataframe

    Parameters
    ----------

    path: 
        path to directory containing parquet files
        
    Returns
    ------- 
    pandas dataframe containing all parquet files in path
    """

    #gets all files in directory and combines them into 1 dataframe  
    directory = Path(path)
    documents = [pd.read_parquet(file) for file in directory.glob('*.parquet')]

    if not documents:
        return pd.DataFrame()

    return pd.concat(documents, ignore_index=True)

def clean_document(document):
    """
    takes a document and cleans it up

    Parameters
    ----------

    document:
        pandas dataframe containing a single document

   Returns
   ------- 
    cleaned pandas dataframe
    """

    #clean out latex @cites and @math and remove spacing
    document['abstract'] = document['abstract'].str.replace(' \n', '', regex=False)
    document['abstract'] = document['abstract'].str.replace(r'@\w+', '', regex=True)
    document['abstract'] = document['abstract'].str.replace(r'\s+', ' ', regex=True)

    document['article'] = document['article'].str.replace(r'@\w+', '', regex=True)
    document['article'] = document['article'].str.replace(r'\s+', ' ', regex=True)

    article_summary = document['article'].str.len().describe()
    abstract_summary = document['abstract'].str.len().describe()

    #remove really long and short papers some are missing and others have many artifacts
    document = document[
        (document['article'].str.len() >= article_summary['25%']) &
        (document['article'].str.len() <= article_summary['75%']) &
        (document['abstract'].str.len() >= abstract_summary['25%']) &
        (document['abstract'].str.len() <= abstract_summary['75%'])
    ]

    document = document.drop_duplicates().reset_index(drop=True)
    
    return document

if __name__ == "__main__":
    with open("config\config.yaml", "r") as f:
        config = yaml.safe_load(f)

    directory_path = config['dataset']['train']

    dataset = load_document(directory_path)
    dataset = clean_document(dataset)

    print(f"Final dataset shape: {dataset.shape}")