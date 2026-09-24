from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pandas as pd
file_path = pd.read_csv('data.csv')
file_path.columns
file_path.dropna(axis=0)
# target
y = file_path.price
# feature
file_feature = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built']
X = file_path[file_feature]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
candidate_max_leaf_nodes = [2, 3, 4, 5]
scores = {}
for cost in candidate_max_leaf_nodes:
    file_model = DecisionTreeRegressor(max_leaf_nodes=cost, random_state=1)
    file_model.fit(train_X, train_y)
    preds = file_model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    scores[cost] = mae
cost_of_house = min(scores, key=scores.get)
print(cost_of_house)
