from sklearn.ensemble import RandomForestRegressor
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
file_model = RandomForestRegressor(random_state=0)
file_model.fit(train_X, train_y)
preds = file_model.predict(val_X)
mae = mean_absolute_error(val_y, preds)
print(mae)
