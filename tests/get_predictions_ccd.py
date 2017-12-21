"""This script should be run on your local machine."""

import requests
import pickle
import numpy as np

API_HOST = 'http://18.217.207.91:5000'

TEST_API = '/test_endpoint'
PREDICT_API = '/predict'
#TRAIN_API = '/train'

DATA_FILE = 'CCD_test_data.dat'

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

def load_data():
    # Load data, test_data file, to server as JSON and convert to
    # list
    test_data = pickle.load(open(DATA_FILE, 'rb'))
    test_data=np.array(test_data).tolist()
    return test_data



def main():
    test_endpoint()
    test_data=load_data()
    predict(test_data)


# Entry point for application (i.e. program starts here)
if __name__ == '__main__':
    main()
