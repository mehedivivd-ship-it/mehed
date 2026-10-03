# House Price Prediction — King County, USA

## 📌 Project Overview
King County (Seattle) house sales data দিয়ে বাড়ির দাম predict করার ML project।

## 📊 Dataset
- **Source:** Kaggle — House Sales in King County, USA
- **Rows:** 4,600 (4,551 after cleaning)
- **Features:** 18 columns
- **Target:** `price`

## 🔍 Approach

### 1. Exploratory Data Analysis (EDA)
- Missing value: 0
- Price skew: 25 → log transform-এ 0.33
- 49টি বাড়ির দাম 0 → ডেটা error, বাদ দিয়েছি

### 2. Data Cleaning
- 0 দামের বাড়ি বাদ
- `street` (4476 unique) বাদ — high cardinality
- `country` (1 unique) বাদ — কোনো তথ্য নেই

### 3. Feature Engineering
- `house_age` = sale_year - yr_built

### 4. Preprocessing (Pipeline)
- **Numerical:** passthrough
- **Categorical:** OneHotEncoder
- **Target:** TransformedTargetRegressor (log1p + expm1)

### 5. Models
- RandomForest (baseline): CV MAE = 111,515
- **XGBoost (final): CV MAE = 102,070**

### 6. Hyperparameter Tuning
- GridSearchCV with 5-fold Cross-Validation

## 📈 Results

| Model | CV MAE |
|---|---|
| Baseline RandomForest | 175,997 |
| + Log transform | 156,621 |
| + Categorical features | 110,696 |
| + house_age | 109,167 |
| **XGBoost (tuned)** | **102,070** |

**Improvement from baseline: 42%**

## 🔑 Key Insights
- `sqft_living` এবং `city` সবচেয়ে গুরুত্বপূর্ণ feature
- Location (city, statezip) = দামের বড় প্রভাবক
- দামি বাড়ি (2 মিলিয়ন+) খারাপ predict হয় — training-এ কম ছিল

## 🛠️ Tools Used
- Python, Pandas, NumPy
- Scikit-learn (Pipeline, ColumnTransformer, GridSearchCV)
- XGBoost
- Matplotlib

## 📁 Files
- `feature.py/intro.py` — Main code (XGBoost, tuned)
- `feature.py/data.csv` — Dataset
- `housepricepredict.py` — Earlier version (RandomForest)

## 👤 Author
**Mehedi** — [GitHub](https://github.com/mehedivd-shp-it)
