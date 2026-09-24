from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
file = pd.read_csv('diabetes.csv')
file.columns
file = file.dropna(how='all')
y = file.Outcome
fetures = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin',
           'BMI', 'DiabetesPedigreeFunction', 'Age']
X = file[fetures]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
file_model = RandomForestRegressor(random_state=0)
file_model.fit(train_X, train_y)
preds = file_model.predict(val_X)
mae = mean_absolute_error(val_y, preds)
print(mae)
