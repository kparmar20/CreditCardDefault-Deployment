"""This script should be run on your local machine."""

import requests
import pickle
import numpy as np

API_HOST = 'http://18.217.207.91:5000'

TEST_API = '/test_endpoint'
PREDICT_API = '/predict'
UPDATE_API = '/update'
#TRAIN_API = '/train'

TEST_DATA=[{'AGE': 29.0,
  'BILL_AMT1': 114538.0,
  'BILL_AMT2': 117171.0,
  'BILL_AMT3': 115941.0,
  'BILL_AMT4': 116877.0,
  'BILL_AMT5': 117359.0,
  'BILL_AMT6': 117029.0,
  'EDUCATION': 2.0,
  'ID': 7091.0,
  'LIMIT_BAL': 120000.0,
  'MARRIAGE': 2.0,
  'PAY_0': 0.0,
  'PAY_2': 0.0,
  'PAY_3': 0.0,
  'PAY_4': 0.0,
  'PAY_5': 0.0,
  'PAY_6': 0.0,
  'PAY_AMT1': 6000.0,
  'PAY_AMT2': 5500.0,
  'PAY_AMT3': 4160.0,
  'PAY_AMT4': 4120.0,
  'PAY_AMT5': 4250.0,
  'PAY_AMT6': 4380.0,
  'SEX': 2.0},
 {'AGE': 27.0,
  'BILL_AMT1': 18094.0,
  'BILL_AMT2': 6942.0,
  'BILL_AMT3': 26855.0,
  'BILL_AMT4': 23287.0,
  'BILL_AMT5': 20885.0,
  'BILL_AMT6': 17595.0,
  'EDUCATION': 3.0,
  'ID': 18872.0,
  'LIMIT_BAL': 80000.0,
  'MARRIAGE': 1.0,
  'PAY_0': 0.0,
  'PAY_2': 0.0,
  'PAY_3': -1.0,
  'PAY_4': 0.0,
  'PAY_5': 0.0,
  'PAY_6': 0.0,
  'PAY_AMT1': 1034.0,
  'PAY_AMT2': 27318.0,
  'PAY_AMT3': 1407.0,
  'PAY_AMT4': 429.0,
  'PAY_AMT5': 417.0,
  'PAY_AMT6': 200.0,
  'SEX': 2.0}]

UPDATE_DATA=[{'AGE': 40.0,
  'BILL_AMT1': 3264.0,
  'BILL_AMT2': 2395.0,
  'BILL_AMT3': 2986.0,
  'BILL_AMT4': 24971.0,
  'BILL_AMT5': 500.0,
  'BILL_AMT6': 2162.0,
  'EDUCATION': 1.0,
  'ID': 8902.0,
  'LIMIT_BAL': 250000.0,
  'MARRIAGE': 2.0,
  'PAY_0': -1.0,
  'PAY_2': -1.0,
  'PAY_3': -1.0,
  'PAY_4': -1.0,
  'PAY_5': -1.0,
  'PAY_6': -1.0,
  'PAY_AMT1': 2395.0,
  'PAY_AMT2': 2991.0,
  'PAY_AMT3': 25185.0,
  'PAY_AMT4': 500.0,
  'PAY_AMT5': 2162.0,
  'PAY_AMT6': 0.0,
  'SEX': 2.0,
  'default.payment.next.month': 0.0},
 {'AGE': 47.0,
  'BILL_AMT1': 1188.0,
  'BILL_AMT2': 2512.0,
  'BILL_AMT3': 3444.0,
  'BILL_AMT4': 1742.0,
  'BILL_AMT5': 502.0,
  'BILL_AMT6': 346.0,
  'EDUCATION': 2.0,
  'ID': 1108.0,
  'LIMIT_BAL': 120000.0,
  'MARRIAGE': 1.0,
  'PAY_0': -1.0,
  'PAY_2': -1.0,
  'PAY_3': -1.0,
  'PAY_4': -1.0,
  'PAY_5': -1.0,
  'PAY_6': -1.0,
  'PAY_AMT1': 2512.0,
  'PAY_AMT2': 3444.0,
  'PAY_AMT3': 1742.0,
  'PAY_AMT4': 502.0,
  'PAY_AMT5': 346.0,
  'PAY_AMT6': 325.0,
  'SEX': 2.0,
  'default.payment.next.month': 0.0},
 {'AGE': 53.0,
  'BILL_AMT1': 35647.0,
  'BILL_AMT2': 34775.0,
  'BILL_AMT3': 35946.0,
  'BILL_AMT4': 37685.0,
  'BILL_AMT5': 33305.0,
  'BILL_AMT6': 28305.0,
  'EDUCATION': 3.0,
  'ID': 29057.0,
  'LIMIT_BAL': 30000.0,
  'MARRIAGE': 1.0,
  'PAY_0': 2.0,
  'PAY_2': 2.0,
  'PAY_3': 0.0,
  'PAY_4': 0.0,
  'PAY_5': 0.0,
  'PAY_6': 0.0,
  'PAY_AMT1': 0.0,
  'PAY_AMT2': 1733.0,
  'PAY_AMT3': 2452.0,
  'PAY_AMT4': 1072.0,
  'PAY_AMT5': 1089.0,
  'PAY_AMT6': 3101.0,
  'SEX': 2.0,
  'default.payment.next.month': 0.0},
 {'AGE': 39.0,
  'BILL_AMT1': 97284.0,
  'BILL_AMT2': 99542.0,
  'BILL_AMT3': 99898.0,
  'BILL_AMT4': 99786.0,
  'BILL_AMT5': 97670.0,
  'BILL_AMT6': 100959.0,
  'EDUCATION': 1.0,
  'ID': 9414.0,
  'LIMIT_BAL': 100000.0,
  'MARRIAGE': 1.0,
  'PAY_0': 2.0,
  'PAY_2': 2.0,
  'PAY_3': 2.0,
  'PAY_4': 2.0,
  'PAY_5': 2.0,
  'PAY_6': 2.0,
  'PAY_AMT1': 4700.0,
  'PAY_AMT2': 2950.0,
  'PAY_AMT3': 3100.0,
  'PAY_AMT4': 5.0,
  'PAY_AMT5': 6170.0,
  'PAY_AMT6': 7.0,
  'SEX': 2.0,
  'default.payment.next.month': 1.0},
 {'AGE': 26.0,
  'BILL_AMT1': 3732.0,
  'BILL_AMT2': 3732.0,
  'BILL_AMT3': 3732.0,
  'BILL_AMT4': 3732.0,
  'BILL_AMT5': 4239.0,
  'BILL_AMT6': 3910.0,
  'EDUCATION': 1.0,
  'ID': 27890.0,
  'LIMIT_BAL': 170000.0,
  'MARRIAGE': 2.0,
  'PAY_0': -1.0,
  'PAY_2': -1.0,
  'PAY_3': -1.0,
  'PAY_4': -1.0,
  'PAY_5': -1.0,
  'PAY_6': -1.0,
  'PAY_AMT1': 3732.0,
  'PAY_AMT2': 3732.0,
  'PAY_AMT3': 3732.0,
  'PAY_AMT4': 4239.0,
  'PAY_AMT5': 3910.0,
  'PAY_AMT6': 4146.0,
  'SEX': 2.0,
  'default.payment.next.month': 0.0}]



def test_endpoint():
    print("Trying test endpoint...")
    url = API_HOST + TEST_API
    print("URL is", url)

    # Try to access the URL. The response will be stored in 'r'.
    r = requests.get(url)

    # The response status code tells us whether or not we were
    # successful in accessing the URL. Generally, HTTP status codes
    # starting with 2 are good and ones starting with 4 or 5 are bad.
    # HTTP status codes:
    # https://en.wikipedia.org/wiki/List_of_HTTP_status_codes
    if r.status_code == 200:
        print("Success for endpoint!")
        print(r.text)
    else:
        print("Status code indicates a problem:", r.status_code)


def predict(data):
    print("Trying predict endpoint...")
    # Note that this is a POST request as we need to send the
    # passenger data to the server.
    # The requests library converts the passenger data into
    # JSON before sending it over. This is because the server
    # expects to receive the passenger data in the form of a JSON.
    r = requests.post(API_HOST + PREDICT_API,
                      json=data)

    # Also note that we're now using r.json(), not r.text.
    # This is because the server sends its response back as a
    # JSON object, which needs to be decoded by the requests
    # library.
    if r.status_code == 200:
        print("Success for predict!")
        print(r.json())
    else:
        print("Status code indicates a problem:", r.status_code)

def update(data):
    print("Trying predict endpoint...")
    # Note that this is a POST request as we need to send the
    # passenger data to the server.
    # The requests library converts the passenger data into
    # JSON before sending it over. This is because the server
    # expects to receive the passenger data in the form of a JSON.
    r = requests.post(API_HOST + UPDATE_API,
                      json=data)

    # Also note that we're now using r.json(), not r.text.
    # This is because the server sends its response back as a
    # JSON object, which needs to be decoded by the requests
    # library.
    if r.status_code == 200:
        print("Success for update!")
        print(r.text)
    else:
        print("Status code indicates a problem:", r.status_code)





def main():
    test_endpoint()
    #predict(TEST_DATA)
    update(UPDATE_DATA)


# Entry point for application (i.e. program starts here)
if __name__ == '__main__':
    main()
