# github repo link for my own reference: https://github.com/rrri01/DATA-221-Group-Project-Raines-Stuff
# group repository link for my own reference: https://github.com/chloeptrsn/DATA-221-Project.git

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
import numpy as np

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

california_house_prices = pd.read_csv('housing.csv', delimiter=',')

#fill in the missing values (the missing values are usually present in total_bedrooms)
california_house_prices["total_bedrooms"] = california_house_prices["total_bedrooms"].fillna(california_house_prices["total_bedrooms"].median())

# mapping (converting categorical to numerical)
mapping = {
    "INLAND": 0,
    "NEAR OCEAN": 1,
    "NEAR BAY": 2,
    "ISLAND": 3,
    "<1H OCEAN": 4
}

california_house_prices["ocean_proximity"] = california_house_prices["ocean_proximity"].map(mapping)


# creates feature matrix X of all columns except "median_house_value" and create label vector y as "median_house_value"
feature_matrix = california_house_prices.drop("median_house_value", axis=1)
target_prices = california_house_prices["median_house_value"]

# splits the data into training data and testing data
x_train, x_test, y_train, y_test = train_test_split(feature_matrix, target_prices, test_size=0.3, random_state=42)

