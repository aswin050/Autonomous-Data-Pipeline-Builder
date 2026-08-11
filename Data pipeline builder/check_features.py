import joblib


features = joblib.load(
    "models/feature_columns.pkl"
)


print("Saved Features:")
print(features)


print("\nNumber of Features:")
print(len(features))