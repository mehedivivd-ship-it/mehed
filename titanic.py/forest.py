from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
file = pd.read_csv('train-selected-columns.csv')
file.columns
file = file.dropna(axis=0)
y = file.Survived
X_features = ['PassengerId', 'Pclass', 'Age', 'Parch']
X = file[X_features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
file_model = RandomForestRegressor(random_state=0)
file_model.fit(train_X, train_y)
file_predict = file_model.predict(val_X)
mae = mean_absolute_error(val_y, file_predict)
print(mae)
