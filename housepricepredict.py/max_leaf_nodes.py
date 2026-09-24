# data import koreci
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
my_file = pd.read_csv('data.csv')
my_file.columns
my_file = my_file.dropna(axis=0)
# target
y = my_file.price
# features
my_file_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                    'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement', 'yr_built']
X = my_file[my_file_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):

    my_model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0)
    my_model.fit(train_X, train_y)
    preds = my_model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    return (mae)


print(get_mae(5, train_X, val_X, train_y, val_y))
print(get_mae(50, train_X, val_X, train_y, val_y))
print(get_mae(100, train_X, val_X, train_y, val_y))
print(get_mae(500, train_X, val_X, train_y, val_y))
