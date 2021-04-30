# Airbnb Price Predictor

![airbnb_logo](Airbnb-Logo.png)

## Introduction

We have all probably used an Airbnb services before, and you might have a question what would my apartment could be worth for a night's stand. With this tool you can find it out! Just enter your property's details and you will get the model's prediction.

First, I scraped Airbnb data for different cities and created a Linear Regression model with L1 Lasso regularization. Using Flask and API and a regular input form was created to access the model for predictions.

## Usage

The project homepage with the prediction form can be found here https://turing24capstone.herokuapp.com. To access API you can call:

- `/predict` - Send a `POST` request to get the model prediction in JSON format.
- `/inferences` - Send a `GET` request to get the last 10 model inferences in JSON format.

At home page, after filling out all the input fields you will get a price prediction with UI.

For an input you should use this request body:

```
{
    "inputs": [
        {
            "city": "Vilnius",
            "country": "Lithuania",
            "apartment_type": "Entire apartment",
            "guests": 4,
            "bedrooms": 1,
            "beds": 2.0,
            "baths": "1 bath",
            "amenities": "[Pool, Wifi, Free parking, Self check-in]",
            "rating": 4.97,
            "reviews": 112.0,
            "is_superhost": 1
        }
    ]
}
```

Model output would look like this:

```
{"Predicted Price": [69.59057843039955]}
```

You can send the requests using a tool like Postman or you can do it with programming language. Here is a Python example:

```
import json
import requests

d = json.dumps({
    "inputs": [
        {
            "city": "Moscow",
            "country": "Russia",
            "apartment_type": "Entire apartment",
            "guests": 2,
            "bedrooms": 1,
            "beds": 4.0,
            "baths": "1.5 bath",
            "amenities": "[Kitchen, Wifi, Pool, Self check-in]",
            "rating": 4.8,
            "reviews": 112.0,
            "is_superhost": 1
        }
    ]
})
resp = requests.post('https://turing24capstone.herokuapp.com/predict', data=d)
print(resp.text)
```

## Modeling

The model and its accompanying methods were created and trained using Jupyter [Notebook](https://github.com/arnoldasjan/AirbnbPricePredictor/blob/master/model/modeling.ipynb) in the Model directory. Model's accuracy is not that great, but in this project the main goal was to create a solid app.

Model's accuracy metrics:

```
Model's R^2 score is: 0.4131447429209173
Mean absolute error is: 23.78837096330827
```

## Future Improvements

In the future, the model could be improved by incorporating images data. Since the look of the property can be really impactful to its price, after learning Deep Learning concepts this project's model's accuracy could be improved by applying Deep Learning methods to listings' images.

## License
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)