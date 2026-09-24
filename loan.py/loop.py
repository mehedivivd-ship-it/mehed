from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import pandas as pd
file = pd.read_csv('train_u6lujuX_CVtuZ9i.csv')
file.columns
file = file.dropna(how='all')
y = file.Loan_Status.map({'N': 0, 'Y': 1})
features = ['ApplicantIncome',
            'CoapplicantIncome', 'Loan_Amount_Term', 'Credit_History']
X = file[features]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
scores = {}
candidate_max_leaf_nodes = [5, 50, 100, 500, 1000, 2000]
for loan in candidate_max_leaf_nodes:
    file_model = DecisionTreeRegressor(
        max_leaf_nodes=loan, random_state=0)
    file_model.fit(train_X, train_y)
    preds = file_model.predict(val_X)
    mae = mean_absolute_error(val_y, preds)
    scores[loan] = mae


print(scores)
