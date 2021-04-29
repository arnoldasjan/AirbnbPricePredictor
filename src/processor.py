import pickle
import json
import pandas as pd
import numpy as np


class Processor:
    """Class to procces input from form and api call, and
    can give a model output.
    """

    def __init__(self):
        self.__reg = pickle.load(open("../model/regression.pkl", "rb"))
        self.__scaler = pickle.load(open("../model/scaler.pkl", "rb"))
        self.__encoder = pickle.load(open("../model/encoder.pkl", "rb"))
        self.__categorical_features = [
            "city",
            "country",
            "apartment_type",
            "baths",
            "amenities",
            "is_superhost",
        ]

    def transform_data(self, df: pd.DataFrame) -> np.array:
        """This method prepares dataframe by encoding categorical features
        and scaling numeric features, then returns prepared data for predictions
        :param df: pandas Dataframe with feature columns
        :return: numpy array with processed features data
        """
        categorical_encoded_data = self.__encoder.transform(
            df[self.__categorical_features].values
        ).toarray()
        scaled_numerical_data = self.__scaler.transform(
            df.drop(self.__categorical_features, axis=1)
        )
        processed_data = np.concatenate(
            (categorical_encoded_data, scaled_numerical_data), axis=1
        )
        return processed_data

    def process_input(self, request_data: str) -> np.array:
        """Loads a post request body as json, converts it to
        features dataframe calls a method that process it and returns
        the result.
        :param request_data: json structured request body
        :return: processed numpy array
        """
        list_of_dicts = json.loads(request_data)["inputs"]
        features = pd.DataFrame(list_of_dicts)
        processed_data = self.transform_data(features)
        return processed_data

    def process_form(self, form_data: dict) -> tuple:
        """Receives a dictionary from input form and processes it
        for the modeling by returning json format input (needed for database insert)
        and processed data as numpy array
        :param form_data: input form dictionary
        :return: tuple of processed data and json type input
        """
        amenities = [
            "Kitchen",
            "Wifi",
            "Free parking",
            "Self check-in",
            "Heating",
            "Pool",
            "Air conditioning",
            "Washer",
        ]
        input_amenities = []
        for amenity in amenities:
            if amenity in form_data:
                input_amenities.append(amenity)

        amenities_str = f"{input_amenities}".replace("'", "")

        input_dict = {
            "city": form_data["city"],
            "country": form_data["country"],
            "apartment_type": form_data["apartmentType"],
            "guests": form_data["guests"],
            "bedrooms": form_data["bedrooms"],
            "beds": form_data["beds"],
            "baths": form_data["baths"],
            "amenities": amenities_str,
            "rating": form_data["rating"],
            "reviews": form_data["reviews"],
            "is_superhost": form_data["superhost"],
        }

        json_input = json.dumps({"inputs": [input_dict]})

        processed_data = self.process_input(json_input)

        return json_input, processed_data

    def predict(self, input_params):
        """Load model and make a predictions and return
        a list of them.
        :return: predictions list
        """
        predictions = self.__reg.predict(input_params)
        predictions = np.maximum(5, predictions)
        return predictions
