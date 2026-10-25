import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------
# 1. Load dataset
# ---------------------------------------

df = pd.read_csv(
    "dataset/TES_100000_requested_variables_only.csv"
)

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# ---------------------------------------
# 2. Separate inputs and target
# ---------------------------------------

X = df.drop(columns=["temperature_profile_C"])

y = df["temperature_profile_C"]

print("\nInput features:", X.shape)
print("Target:", y.shape)

print("\nInput columns:")
print(X.columns.tolist())


# ---------------------------------------
# 3. Train-Test Split
# ---------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain data:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nTest data:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# ---------------------------------------
# 4. Feature Scaling
# ---------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

print("\nScaling completed!")

print("Scaled training data shape:",
      X_train_scaled.shape)

print("Scaled testing data shape:",
      X_test_scaled.shape)