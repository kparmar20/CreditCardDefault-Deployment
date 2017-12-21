import time
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
import numpy as np


MODEL_DIRECTORY = 'model'
MODEL_FILE_NAME = '%s/model_CCD_file.pkl' % MODEL_DIRECTORY

def train(df):
    df_ = df[include]
    print("Training data sample:\n", df_.head())
    categoricals = []  # going to one-hot encode categorical variables

    for col, col_type in df_.dtypes.items():
        if col_type == 'O':
            categoricals.append(col)
        else:
            df_[col].fillna(0, inplace=True)  # fill NA's with 0 for ints/floats, too generic

    # get_dummies effectively creates one-hot encoded variables
    df_ohe = pd.get_dummies(df_, columns=categoricals, dummy_na=True)

    x = df_ohe[df_ohe.columns.difference([dependent_variable])]
    y = df_ohe[dependent_variable]

    # capture a list of columns that will be used for prediction
    model_columns = list(x.columns)
    print("Model columns are", model_columns)

    model = rf()
    start = time.time()
    model.fit(x, y)
    print('Trained in %.1f seconds' % (time.time() - start))
    print('Model training score: %s' % model.score(x, y))

    return model_columns, model


def predict(input_df, model):
    input_array=np.array(input_df) 
    print(type(model))   
    print("predict in model utils")
    predictions = model.predict(input_array)

    return predictions.tolist() 




