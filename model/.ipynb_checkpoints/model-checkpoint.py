import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import Lasso

CATEGORICAL_FEATURES = ['city', 'country', 'apartment_type', 'bedrooms', 'baths', 'amenities', 'is_superhost']

def preprocessing():
    onehotencoder = OneHotEncoder(handle_unknown="ignore")
    categorical_encoded_data = onehotencoder.fit_transform(features[CATEGORICAL_FEATURES].values).toarray()

    scaler = StandardScaler()
    scaled_numerical_data = scaler.fit_transform(features.drop(CATEGORICAL_FEATURES, axis=1))

    processed_data = np.concatenate((categorical_encoded_data, scaled_numerical_data), axis=1)


data = pd.read_csv('airbnb_data.csv')

features = data.drop(['price', 'listing_url', 'image_url', 'title', 'district'], axis=1)
target = data['price']

features['rating'].fillna(features['rating'].mean(), inplace=True)
features['reviews'].fillna(1, inplace=True)
features['baths'].fillna('1 bath', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(processed_data, target, test_size=0.3)
reg = Lasso()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)
print('Regression score of the model', reg.score(X_test, y_test))
print('Mean absolute error for the model', mean_absolute_error(y_test, pred))