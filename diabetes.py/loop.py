from sklearn.tree import DecisionTreeRegressor
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
scores = {}
candidate_max_leaf_nodes = [5, 50, 100, 500, 5000]
for diabetes in candidate_max_leaf_nodes:
    file_model = DecisionTreeRegressor(
        max_leaf_nodes=diabetes, random_state=0)
    file_model.fit(train_X, train_y)
    preds = file_model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    scores[diabetes] = mae


print(scores)
