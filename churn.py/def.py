from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd
file = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
file.columns
file = file.dropna(axis=0)
y = file.Churn.map({'No': 0, 'Yes': 1})
features = ['tenure', 'MonthlyCharges']
X = file[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)


def churn(max_leaf_nodes, train_X, val_X, train_y, val_y):
    file_model = DecisionTreeRegressor(
        max_leaf_nodes=max_leaf_nodes, random_state=0)
    file_model.fit(train_X, train_y)
    preds = file_model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    return (mae)


print(churn(5, train_X, val_X, train_y, val_y))
print(churn(50, train_X, val_X, train_y, val_y))
print(churn(100, train_X, val_X, train_y, val_y))
print(churn(200, train_X, val_X, train_y, val_y))
print(churn(500, train_X, val_X, train_y, val_y))
