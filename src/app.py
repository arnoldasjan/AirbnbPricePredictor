from flask import Flask, render_template, request
import json
from database.database_operations import insert_inference, get_inferences
from datetime import date, datetime
from src.processor import Processor

app = Flask(__name__, template_folder="../templates")


def json_serial(obj):
    """JSON serializer for objects not serializable
    by default json code
    """
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError("Type %s not serializable" % type(obj))


@app.route("/", methods=["GET", "POST"])
def home():
    """Homepage that has an input form where you can enter data
    and it returns the prediction of a listing. From this page you can
    also go to check last 10 inferences
    :return: html template based on GET or POST request
    """
    if request.method == "POST":
        try:
            processor = Processor()
            processed_form = processor.process_form(request.form.to_dict())
            json_input, input_params = processed_form
            prediction = processor.predict(input_params)
            insert_inference(json_input, json.dumps(prediction.tolist()))
            return render_template("prediction.html", prediction=prediction)
        except:
            return "Something went wrong, please try again later", 400
    else:
        return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Predict url is for an API calls to predict the price of
    a listing and returns json format prediction object
    :return: json format price prediction
    """
    try:
        processor = Processor()
        input_params = processor.process_input(request.data)
        predictions = processor.predict(input_params)
        json_input = json.loads(request.data)
        insert_inference(json.dumps(json_input), json.dumps(predictions.tolist()))
        return json.dumps({"Predicted Price": predictions.tolist()})
    except (KeyError, json.JSONDecodeError, AssertionError, ValueError):
        return json.dumps({"error": "CHECK INPUT"}), 400
    except Exception as err:
        return json.dumps({"error": "PREDICTION FAILED", "message": {err}}), 500


@app.route("/inferences", methods=["GET"])
def inferences():
    """Page selects from database and returns
    the 10 most recent inferences in a html format table
    :return: html template with table of recent inferences
    """
    try:
        infers = get_inferences()
        return json.dumps({"predictions": infers}, default=json_serial)
    except Exception as err:
        return json.dumps({"error": "COULD NOT GET INFERENCES", "message": {err}}), 500


if __name__ == "__main__":
    app.run()
