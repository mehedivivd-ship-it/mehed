from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import TransformedTargetRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib

# ১. ডেটা লোড
file = pd.read_csv('data.csv')
file = file[file['price'] > 0]
file['date'] = pd.to_datetime(file['date'])
file['house_age'] = file['date'].dt.year - file['yr_built']
# ৮. সব কলামের unique count
# for col in file.columns:
#     print(f"{col}: {file[col].nunique()} unique, dtype={file[col].dtype}")

# ৫. X, y বানাও
y = file['price']    # ← raw price, log না
no_need = ['price', 'street', 'country', 'date',]
X = file.drop(no_need, axis=1)

# ৬. Split (raw y দিয়ে)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)
categorical_cols = [
    cname for cname in X_train.columns if X_train[cname].dtype == "object" or X_train[cname].dtype == "str"]
numerical_col = [cname for cname in X_train.columns if X_train[cname].dtype in [
    'int64', 'float64']]
preprocessor = ColumnTransformer(transformers=[('num', 'passthrough', numerical_col), (
    'cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)])


# ৭. Model — TransformedTargetRegressor দিয়ে
model = TransformedTargetRegressor(
    regressor=XGBRegressor(random_state=0,
                           n_estimators=200,
                           learning_rate=0.1,
                           n_jobs=-1,
                           max_depth=6,
                           verbosity=0),
    func=np.log1p,
    inverse_func=np.expm1
)
my_pipeline = Pipeline(
    steps=[('preprocessor', preprocessor), ('model', model)])
scores = cross_val_score(my_pipeline, X, y, cv=5,
                         scoring='neg_mean_absolute_error')
print("Scores:", -scores)
print(-scores.mean())
print("Std:", scores.std())
param_grid = {
    'model__regressor__n_estimators': [50, 100, 200],
    'model__regressor__max_depth': [5, 10, 20, None]
}
grid_search = GridSearchCV(
    my_pipeline,
    param_grid,
    cv=5,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)
print("beast param:", grid_search.best_params_)
print("Best CV MAE:", grid_search.best_score_)
best_pipeline = grid_search.best_estimator_
scores = cross_val_score(
    best_pipeline, X, y, scoring='neg_mean_absolute_error', n_jobs=-1)
print(-scores)
print(-scores.mean())
print(scores.std())
best_model = grid_search.best_estimator_
preprocessor = best_model.named_steps['preprocessor']
cat_encoder = preprocessor.named_transformers_['cat']
cat_features = cat_encoder.get_feature_names_out(categorical_cols)
all_feature = list(numerical_col) + list(cat_features)
rf = best_model.named_steps['model'].regressor_
importances = rf.feature_importances_
importances_df = pd.DataFrame({
    'feature': all_feature,
    'importance': importances
}).sort_values('importance', ascending=False)
print(importances_df.head(20))
preds = grid_search.predict(X_val)
fig, axis = plt.subplots(1, 2, figsize=(14, 6))
axis[0].scatter(y_val, preds, alpha=0.5, s=10)
axis[0].plot([y_val.min(), y_val.max()], [
             y_val.min(), y_val.max()], 'r--', lw=2)
axis[0].set_xlabel('Actual Price')
axis[0].set_ylabel('predicted price')
axis[0].set_title('Actual vs Predicted')
axis[0].grid(True, alpha=0.3)
residuals = y_val - preds
axis[1].scatter(y_val, residuals, alpha=0.5, s=10)
axis[1].axhline(y=0, color='r', linestyle='--', lw=2)
axis[1].set_xlabel('predicted price')
axis[1].set_ylabel('residuals')
axis[1].set_title('residuals plot')
axis[1].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print("Mean residual:", residuals.mean())
print("Std residual:", residuals.std())
print("Max residual:", residuals.max())
print("Min residual:", residuals.min())
# ৮. MAE
mae = mean_absolute_error(y_val, preds)
print("MAE (raw scale):", mae)
joblib.dump(grid_search.best_estimator_, 'house_price_xgboost.pkl')
print("model saved")
loaded_model = joblib.load('house_price_xgboost.pkl')
test_preds = loaded_model.predict(X_val[:5])
print("final:", test_preds)
