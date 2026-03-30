# github repo link for my own reference: https://github.com/rrri01/DATA-221-Group-Project-Raines-Stuff
# group repository link for my own reference: https://github.com/chloeptrsn/DATA-221-Project.git

import pandas as pd

housing_data = pd.read_csv('housing.csv', delimiter=',')
print(housing_data)

# the next few lines were just to figure out what the categorical options were
# d = {}
# l = []
# for i in housing_data["ocean_proximity"]:
#     l.append(i)
#
# for x in l:
#     d[x] = 0
# print(d)

# target: housing_data["median_house_value"]
# categorical: ocean_proximity
# CATEGORICAL OPTIONS:
# NEAR BAY
# <1H OCEAN
# INLAND
# NEAR OCEAN
# ISLAND

from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

housing_data = housing_data.replace({"NEAR BAY": 0, "<1H OCEAN": 1, "INLAND": 2, "NEAR OCEAN": 3, "ISLAND": 4})

print(housing_data)

