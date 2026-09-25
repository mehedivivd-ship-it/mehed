import xgboost
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
file = pd.read_csv('train.csv')
file.columns
no_need = ['SalePrice', 'Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence']
X = file.drop(no_need, axis=1)
y = file.SalePrice
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=0)
categorical_cols = [
    cname for cname in X_train.columns if X_train[cname].dtype == "object"]
numerical_cols = [
    cname for cname in X_train.columns if X_train[cname].dtype in ['int64', 'float64']]
numerical_transformer = SimpleImputer(strategy='constant')
categorical_transformer = Pipeline(steps=[('sim', SimpleImputer(
    strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(transformers=[(
    'num', numerical_transformer, numerical_cols), ('cat', categorical_transformer, categorical_cols)])
model = XGBRegressor(n_estimators=100, random_state=0,
                     n_jobs=-1, learning_rate=0.05)
my_pipeline = Pipeline(steps=[('preproessor', preprocessor), ('model', model)])
score = -1 * cross_val_score(my_pipeline, X, y,
                             scoring='neg_mean_absolute_error')
my_pipeline.fit(X_train, y_train)
preds = my_pipeline.predict(X_valid)
mae = mean_absolute_error(y_valid, preds)
print(mae)
