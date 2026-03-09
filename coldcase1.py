import pandas as pd
import numpy as np
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, precision_score, recall_score

CSV_PATH = r"C:\Users\chamu\OneDrive\Desktop\particum\chicago_crime_full.csv"

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV path not found at {CSV_PATH}")

print("Loading dataset...")
df = pd.read_csv(CSV_PATH, low_memory=False)

# --- Column normalization
df.columns = [c.lower().replace(' ', '_') for c in df.columns]

# --- Cleaning
if 'primary_type' in df.columns:
    df['primary_type'] = df['primary_type'].replace('CRIM SEXUAL ASSAULT', 'CRIMINAL SEXUAL ASSAULT')

for col in ['latitude', 'longitude']:
    if col in df.columns and 'district' in df.columns:
        median_val = df.groupby('district')[col].transform('median')
        df[col] = df[col].fillna(median_val)

critical_cols = ['latitude', 'longitude', 'district', 'ward', 'community_area']
df.dropna(subset=[c for c in critical_cols if c in df.columns], inplace=True)

# --- Filter violent crimes + target
violent_crime_types = [
    "CRIMINAL SEXUAL ASSAULT", "ASSAULT", "SEX OFFENSE", "STALKING",
    "KIDNAPPING", "ROBBERY", "BATTERY", "HOMICIDE", "ARSON",
    "HUMAN TRAFFICKING", "CRIMINAL TRESPASS", "OFFENSE INVOLVING CHILDREN"
]

violent_crimes_df = df[df['primary_type'].isin(violent_crime_types)].copy()
violent_crimes_df['date'] = pd.to_datetime(violent_crimes_df['date'], errors='coerce')
violent_crimes_df.dropna(subset=['date'], inplace=True)

# Ensure arrest is boolean (important!)
if violent_crimes_df["arrest"].dtype != bool:
    violent_crimes_df["arrest"] = violent_crimes_df["arrest"].astype(str).str.lower().map({"true": True, "false": False})

max_dt = violent_crimes_df['date'].max()
violent_crimes_df['case_age_days'] = (max_dt - violent_crimes_df['date']).dt.days

violent_crimes_df['cold_case'] = (
    (violent_crimes_df['arrest'] == False) &
    (violent_crimes_df['case_age_days'] > 365)
).astype(int)

drop_cols = ['cold_case', 'arrest', 'date', 'id', 'case_number', 'updated_on', 'location']
X = violent_crimes_df.drop(columns=[c for c in drop_cols if c in violent_crimes_df.columns])
y = violent_crimes_df['cold_case']

# --- Time-based split (by age sorting)
X = X.sort_values('case_age_days', ascending=False)
y = y.loc[X.index]

split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- Preprocess
categorical_features = ['primary_type', 'description', 'location_description', 'domestic']
numerical_features = ['beat', 'district', 'latitude', 'longitude', 'case_age_days']

string_caster = FunctionTransformer(lambda x: x.astype(str))

cat_pipe = Pipeline([
    ('caster', string_caster),
    ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, [f for f in numerical_features if f in X.columns]),
    ('cat', cat_pipe, [f for f in categorical_features if f in X.columns])
])

# --- Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear', class_weight='balanced'),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1, random_state=42, class_weight='balanced'),
    'HistGradient Boosting': HistGradientBoostingClassifier(random_state=42, max_iter=200, max_depth=10)
}

results = []
pipelines = {}

for name, model in models.items():
    print(f"Training {name}...")
    pipe = Pipeline([('pre', preprocessor), ('clf', model)])
    pipe.fit(X_train, y_train)
    pipelines[name] = pipe

    probs = pipe.predict_proba(X_test)[:, 1]
    pred = (probs > 0.5).astype(int)

    p, r, _ = precision_recall_curve(y_test, probs)
    results.append({
        'Model': name,
        'ROC-AUC': roc_auc_score(y_test, probs),
        'PR-AUC': auc(r, p),
        'Precision': precision_score(y_test, pred),
        'Recall': recall_score(y_test, pred)
    })

res_df = pd.DataFrame(results).sort_values('PR-AUC', ascending=False)
print("\n--- Final Model Comparison ---")
print(res_df)

# --- Select best pipeline
best_model_name = res_df.iloc[0]["Model"]
best_pipeline = pipelines[best_model_name]
print("\nBest model selected:", best_model_name)

# --- Single-case predict
def predict_cold_for_case(case_dict_or_series, pipeline, feature_columns, threshold=0.35):
    if isinstance(case_dict_or_series, pd.Series):
        row = case_dict_or_series.to_frame().T
    else:
        row = pd.DataFrame([case_dict_or_series])

    for col in feature_columns:
        if col not in row.columns:
            row[col] = np.nan
    row = row[feature_columns]

    prob = pipeline.predict_proba(row)[:, 1][0]
    return {
        "cold_probability": float(prob),
        "cold_prediction": int(prob >= threshold),
        "threshold_used": threshold
    }

feature_columns = list(X_train.columns)
example_case = X_test.iloc[0]
print("\nExample case prediction:", predict_cold_for_case(example_case, best_pipeline, feature_columns, threshold=0.35))

from datetime import datetime

def predict_target_case(target_case, pipeline, feature_columns, threshold=0.35):

    case = target_case.copy()

    # Convert date
    case_date = pd.to_datetime(case['date'])

    # Calculate case age
    today = pd.Timestamp.now()
    case['case_age_days'] = (today - case_date).days

    # Convert to dataframe
    row = pd.DataFrame([case])

    # Ensure correct columns
    for col in feature_columns:
        if col not in row.columns:
            row[col] = np.nan

    row = row[feature_columns]

    # Predict probability
    prob = pipeline.predict_proba(row)[0][1]

    prediction = int(prob >= threshold)

    return {
        "cold_case_probability": float(prob),
        "cold_case_prediction": prediction,
        "threshold": threshold
    }

target_case = {
    'id': 10000000,
    'date': '2026-02-11 00:00:00',
    'primary_type': 'OFFENSE INVOLVING CHILDREN',
    'description': 'CHILD ABDUCTION',
    'location_description': 'ALLEY',
    'latitude': 41.794173,
    'longitude': -87.703576,
    'arrest': False,
    'domestic': False,
    'beat': 923,
    'district': 9
}

feature_columns = list(X_train.columns)

result = predict_target_case(
    target_case,
    best_pipeline,
    feature_columns,
    threshold=0.35
)

print(result)