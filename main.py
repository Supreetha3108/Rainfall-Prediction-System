import pandas as pd

# =========================
# LOAD DATASET
# =========================
df = pd.read_csv("data/weatherAUS.csv")

# =========================
# REMOVE HIGH MISSING COLUMNS
# =========================
df.drop(
    ['Evaporation', 'Sunshine', 'Cloud9am', 'Cloud3pm'],
    axis=1,
    inplace=True
)

# Remove Date column
df.drop('Date', axis=1, inplace=True)

# =========================
# HANDLE MISSING VALUES
# =========================

# Numerical → Mean
numerical_cols = df.select_dtypes(
    include=['float64', 'int64']
).columns

for col in numerical_cols:
    df[col] = df[col].fillna(df[col].mean())

# Categorical → Mode
categorical_cols = df.select_dtypes(
    include=['object']
).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# =========================
# ENCODE CATEGORICAL DATA
# =========================
df = pd.get_dummies(
    df,
    drop_first=True
)

# =========================
# FEATURES & TARGET
# =========================
X = df.drop(
    "RainTomorrow_Yes",
    axis=1
)

y = df["RainTomorrow_Yes"]

print("\nFeature Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nTarget Distribution:")
print(y.value_counts())

# =========================
# TRAIN TEST SPLIT
# =========================
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

# =========================
# LOGISTIC REGRESSION MODEL
# =========================
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(
    max_iter=500,
    solver="saga"
)

model.fit(
    X_train,
    y_train
)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(
    X_test
)

# =========================
# EVALUATION
# =========================
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nModel Accuracy:")
print(accuracy)