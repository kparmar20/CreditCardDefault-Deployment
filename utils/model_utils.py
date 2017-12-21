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
    #print(type(model))   
    print("predict in model utils")
    predictions = model.predict(input_array)

    return predictions.tolist() 

def transform(dict_in):
    #transform data input from dictionary and transform for model input

    df=pd.DataFrame.from_dict(dict_in)
    df['LIMIT_BAL'] = (df['LIMIT_BAL'] - df['LIMIT_BAL'].mean()) / df['LIMIT_BAL'].std()
    df['AGE'] = (df['AGE'] - df['AGE'].mean()) / df['AGE'].std()
    df['BILL_AMT1'] = (df['BILL_AMT1'] - df['BILL_AMT1'].mean()) / df['BILL_AMT1'].std()
    df['BILL_AMT2'] = (df['BILL_AMT2'] - df['BILL_AMT2'].mean()) / df['BILL_AMT2'].std()
    df['BILL_AMT3'] = (df['BILL_AMT3'] - df['BILL_AMT3'].mean()) / df['BILL_AMT3'].std()
    df['BILL_AMT4'] = (df['BILL_AMT4'] - df['BILL_AMT4'].mean()) / df['BILL_AMT4'].std()
    df['BILL_AMT5'] = (df['BILL_AMT5'] - df['BILL_AMT5'].mean()) / df['BILL_AMT5'].std()
    df['BILL_AMT6'] = (df['BILL_AMT6'] - df['BILL_AMT6'].mean()) / df['BILL_AMT6'].std()
    df['PAY_AMT1'] = (df['PAY_AMT1'] - df['PAY_AMT1'].mean()) / df['PAY_AMT1'].std()
    df['PAY_AMT2'] = (df['PAY_AMT2'] - df['PAY_AMT2'].mean()) / df['PAY_AMT2'].std()
    df['PAY_AMT3'] = (df['PAY_AMT3'] - df['PAY_AMT3'].mean()) / df['PAY_AMT3'].std()
    df['PAY_AMT4'] = (df['PAY_AMT4'] - df['PAY_AMT4'].mean()) / df['PAY_AMT4'].std()
    df['PAY_AMT5'] = (df['PAY_AMT5'] - df['PAY_AMT5'].mean()) / df['PAY_AMT5'].std()
    df['PAY_AMT6'] = (df['PAY_AMT6'] - df['PAY_AMT6'].mean()) / df['PAY_AMT6'].std()
    df=df.dropna()
    df.drop('ID', axis=1, inplace=True)
    print("Input transformed for model")    

    return df

def transform_update(dict_in_update):
    #transform data for update/incremental data training of SGD model

    df1=pd.DataFrame.from_dict(dict_in_update)
    df1['LIMIT_BAL'] = (df1['LIMIT_BAL'] - df1['LIMIT_BAL'].mean()) / df1['LIMIT_BAL'].std()
    df1['AGE'] = (df1['AGE'] - df1['AGE'].mean()) / df1['AGE'].std()
    df1['BILL_AMT1'] = (df1['BILL_AMT1'] - df1['BILL_AMT1'].mean()) / df1['BILL_AMT1'].std()
    df1['BILL_AMT2'] = (df1['BILL_AMT2'] - df1['BILL_AMT2'].mean()) / df1['BILL_AMT2'].std()
    df1['BILL_AMT3'] = (df1['BILL_AMT3'] - df1['BILL_AMT3'].mean()) / df1['BILL_AMT3'].std()
    df1['BILL_AMT4'] = (df1['BILL_AMT4'] - df1['BILL_AMT4'].mean()) / df1['BILL_AMT4'].std()
    df1['BILL_AMT5'] = (df1['BILL_AMT5'] - df1['BILL_AMT5'].mean()) / df1['BILL_AMT5'].std()
    df1['BILL_AMT6'] = (df1['BILL_AMT6'] - df1['BILL_AMT6'].mean()) / df1['BILL_AMT6'].std()
    df1['PAY_AMT1'] = (df1['PAY_AMT1'] - df1['PAY_AMT1'].mean()) / df1['PAY_AMT1'].std()
    df1['PAY_AMT2'] = (df1['PAY_AMT2'] - df1['PAY_AMT2'].mean()) / df1['PAY_AMT2'].std()
    df1['PAY_AMT3'] = (df1['PAY_AMT3'] - df1['PAY_AMT3'].mean()) / df1['PAY_AMT3'].std()
    df1['PAY_AMT4'] = (df1['PAY_AMT4'] - df1['PAY_AMT4'].mean()) / df1['PAY_AMT4'].std()
    df1['PAY_AMT5'] = (df1['PAY_AMT5'] - df1['PAY_AMT5'].mean()) / df1['PAY_AMT5'].std()
    df1['PAY_AMT6'] = (df1['PAY_AMT6'] - df1['PAY_AMT6'].mean()) / df1['PAY_AMT6'].std()
    df1=df1.dropna()
    df1.drop('ID', axis=1, inplace=True)
    y=df1['default.payment.next.month']
    X=df1.drop(['default.payment.next.month'],axis=1)

    return X,y

def update(model,X, y):
    print("update model in model utils")
    model.partial_fit(X,y)

