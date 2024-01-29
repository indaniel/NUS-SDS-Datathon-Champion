# Model Loading/Downloading

Models: https://drive.google.com/drive/folders/1P_sAN-WML2i3Z_vNZL0Pu-Q2Nj2nXKed?usp=drive_link

1) Download all pickle files from Google Drive
2) Download it to main folder.
3) The notebook function `testing_hidden_data` should automatically load the model. If not, the following code will help

# Loading of models:
```python
#Imports
import pandas as pd
import numpy as np
import string
import nltk
import reverse_geocoder as rg
import joblib
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def testing_hidden_data(hidden_data: pd.DataFrame) -> list:
    '''DO NOT REMOVE THIS FUNCTION.

    The function accepts a dataframe as input and return an iterable (list)
    of binary classes as output.

    The function should be coded to test on hidden data
    and should include any preprocessing functions needed for your model to perform. 
        
    All relevant code MUST be included in this function.'''

    #Install NLTK packages
    nltk.download('popular')   

    #Load Pipeline and Model from pickle
    processData = joblib.load('processData.pkl')
    model = joblib.load('model.pkl')

    #Preprocess data and predict
    X_pred = processData.transform(hidden_data)
    y_pred = model.predict(X_pred)
    result = y_pred.tolist()

    return result
```


# Expected Functions that are required and must be entered/run in the notebook (for SKLearn Pipeline to function)
```python
#Include Functions used in Pipeline
def dropColumns(data, columns):
    return data.drop(columns, axis=1)

def copy(data):
    return data

#"One Hot Encode" Import/Export Status
def handleImportExport(data):
    data['Import'] = data['Import/Export Status'].apply(lambda x: 1 if x == 'Import' or x == 'Both Imports & Exports' else 0)
    data['Export'] = data['Import/Export Status'].apply(lambda x: 1 if x == 'Export' or x == 'Both Imports & Exports' else 0)
    return data.drop(['Import/Export Status'], axis=1)

#"One Hot Encode" Entity Type
def handleEntityType(data):
    data['Entity_Parent'] = data['Entity Type'].apply(lambda x: 1 if x == 'Parent' else 0)
    data['Entity_Subsidiary'] = data['Entity Type'].apply(lambda x: 1 if x == 'Subsidiary' else 0)
    data['Entity_Branch'] = data['Entity Type'].apply(lambda x: 1 if x == 'Branch' else 0)
    data['Entity_Independent'] = data['Entity Type'].apply(lambda x: 1 if x == 'Independent' else 0)
    data['Entity_Others'] = data['Entity Type'].apply(lambda x: 1 if x not in ['Parent', 'Subsidiary', 'Branch', 'Independent'] else 0)
    return data.drop(['Entity Type'], axis=1)

#Extract and Average imputations within the pipeline
def handleImputation(data):
    #split ndarray into 3 by columns
    original, imputeKNN, imputeRegressor = np.hsplit(data,3)
    #Fill null/missing values in original with the average of the imputations.
    original = np.where(np.isnan(original), ((imputeKNN + imputeRegressor) / 2).astype(int), original)
    return original

#Cleans text data by filling null values, merging columns, tokenizing, removing punctuation, stopwords, and stemming.
def textCleaning(text):
    text = text.fillna('') 
    text['text'] = text.apply(lambda x: ' '.join(x), axis=1) 
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words('english'))

    def clean_text(text):
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word not in string.punctuation]
        tokens = [word for word in tokens if word not in stop_words]
        tokens = [stemmer.stem(word) for word in tokens]
        return ' '.join(tokens)

    text['text'] = text['text'].apply(clean_text)
    return text['text']```
