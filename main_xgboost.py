import pandas as pd

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv("data/weatherAUS.csv")

# ==========================
# REMOVE USELESS COLUMNS
# ==========================

df.drop(
    ['Evaporation', 'Sunshine', 'Cloud3pm', 'Cloud9am'],
    axis=1,
    inplace=True
)

df.drop('Date', axis=1, inplace=True)

# ==========================
# HANDLE MISSING VALUES
# ==========================

numerical_cols = df.select_dtypes(
    include=['float64', 'int64']
).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].mean())

categorical_cols = df.select_dtypes(
    include=['object']
).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# ==========================
# CONVERT TEXT TO NUMBERS
# ==========================

df = pd.get_dummies(
    df,
    drop_first=True
)

# ==========================
# SELECT IMPORTANT FEATURES
# ==========================

selected_features = [
    'MinTemp',
    'MaxTemp',
    'Rainfall',
    'WindGustSpeed',
    'Humidity9am',
    'Humidity3pm',
    'Pressure9am',
    'Pressure3pm'
]

X = df[selected_features]

y = df['RainTomorrow_Yes']

# ==========================
# TRAIN TEST SPLIT
# ==========================

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# ==========================
# XGBOOST MODEL
# ==========================

from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=150,
    max_depth=5,
    learning_rate=0.05,
    random_state=42
)

model.fit(
    X_train,
    y_train
)

# ==========================
# PREDICTION
# ==========================

y_pred = model.predict(X_test)

# ==========================
# ACCURACY
# ==========================

from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nXGBoost Accuracy:")
print(round(accuracy * 100, 2), "%")

# ==========================
# SAVE MODEL
# ==========================

import pickle

pickle.dump(
    model,
    open("models/rainfall_model.pkl", "wb")
)

print("\nModel Saved Successfully!")