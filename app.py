import pandas as pd
from flask import Flask, render_template, request
import pickle
import json
import numpy as np
from database.database_operations import insert_inference, get_inferences

app = Flask(__name__)

SAVED_MODEL_PATH = "model/regression.pkl"
SAVED_ENCODER_PATH = "model/encoder.pkl"
SAVED_SCALER_PATH = "model/scaler.pkl"
CATEGORICAL_FEATURES = ['city', 'country', 'apartment_type', 'baths', 'amenities', 'is_superhost']

reg = pickle.load(open(SAVED_MODEL_PATH, "rb"))
scaler = pickle.load(open(SAVED_SCALER_PATH, "rb"))
encoder = pickle.load(open(SAVED_ENCODER_PATH, "rb"))


def __prepare_data(df: pd.DataFrame) -> np.array:
    """ This method prepares dataframe by encoding categorical features
    and scaling numeric features, then returns prepared data for predictions
    :param df: pandas Dataframe with feature columns
    :return: numpy array with processed features data
    """
    categorical_encoded_data = encoder.transform(df[CATEGORICAL_FEATURES].values).toarray()
    scaled_numerical_data = scaler.transform(df.drop(CATEGORICAL_FEATURES, axis=1))
    processed_data = np.concatenate((categorical_encoded_data, scaled_numerical_data), axis=1)
    return processed_data


def __process_input(request_data: str) -> np.array:
    """Loads a post request body as json, converts it to
    features dataframe calls a method that process it and returns
    the result.
    :param request_data: json structured request body
    :return: processed numpy array
    """
    list_of_dicts = json.loads(request_data)["inputs"]
    features = pd.DataFrame(list_of_dicts)
    processed_data = __prepare_data(features)
    return processed_data


def __process_form(form_data: dict) -> tuple:
    """Receives a dictionary from input form and processes it
    for the modeling by returning json format input (needed for database insert)
    and processed data as numpy array
    :param form_data: input form dictionary
    :return: tuple of processed data and json type input
    """
    try:
        amenities = ['Kitchen', 'Wifi', 'Free parking', 'Self check-in', 'Heating',
                     'Pool', 'Air conditioning', 'Washer']
        input_amenities = []
        for amenity in amenities:
            if amenity in form_data:
                input_amenities.append(amenity)

        amenities_str = f"{input_amenities}".replace('\'', '')

        input_dict = {
            "city": form_data['city'],
            "country": form_data['country'],
            "apartment_type": form_data['apartmentType'],
            "guests": form_data['guests'],
            "bedrooms": form_data['bedrooms'],
            "beds": form_data['beds'],
            "baths": form_data['baths'],
            "amenities": amenities_str,
            "rating": form_data['rating'],
            "reviews": form_data['reviews'],
            "is_superhost": form_data['superhost']
        }

        json_input = json.dumps({"inputs": [input_dict]})

        processed_data = __process_input(json_input)
    except Exception as err:
        print(err)

    return processed_data, json_input


@app.route('/', methods=['GET', 'POST'])
def home():
    """Homepage that has an input form where you can enter data
    and it returns the prediction of a listing. From this page you can
    also go to check last 10 inferences
    :return: html template based on GET or POST request
    """
    if request.method == 'POST':
        try:
            processed_form = __process_form(request.form.to_dict())
            input_params = processed_form[0]
            json_input = processed_form[1]
            prediction = reg.predict(input_params)
            prediction = np.maximum(5, prediction)
            insert_inference(json_input, json.dumps(prediction.tolist()))
            return render_template('prediction.html', prediction=prediction)
        except Exception as err:
            print(err)
            return 'Something went wrong, please try again later'
    else:
        return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Predict url is for an API calls to predict the price of
    a listing and returns json format prediction object
    :return: json format price prediction
    """
    try:
        input_params = __process_input(request.data)
        json_input = json.loads(request.data)
        predictions = reg.predict(input_params)
        predictions = np.maximum(5, predictions)
        insert_inference(json.dumps(json_input), json.dumps(predictions.tolist()))
        return json.dumps({"Predicted Price": predictions.tolist()})
    except (KeyError, json.JSONDecodeError, AssertionError):
        return json.dumps({"error": "CHECK INPUT"}), 400
    except Exception as err:
        print(err)
        return json.dumps({"error": "PREDICTION FAILED", "message": {err}}), 500


@app.route('/inferences', methods=['GET'])
def inferences():
    """Page selects from database and returns
    the 10 most recent inferences in a html format table
    :return: html template with table of recent inferences
    """
    try:
        table = get_inferences()
        return render_template('inferences.html',  tables=[table.to_html(classes='table', header="true", index=False)])
    except Exception as err:
        print(err)
        return 'Could not load last inferences, sorry!'


if __name__ == '__main__':
    app.run()
