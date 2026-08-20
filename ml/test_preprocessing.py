from preprocessing import (
    load_data,
    inspect_data,
    separate_target,
    remove_id_columns,
    remove_high_missing_columns,
    identify_categorical_columns,
    remove_constant_columns,
    encode_categorical_columns,
    split_data
)

from feature_engineering import (
    extract_time_features,
    extract_description_features,
    extract_location_features,
    extract_business_name_features
)

from feature_engineering import extract_time_features
data_path = "data/fraud_dataset.csv"

df = load_data(data_path)

print("Dataset loaded successfully!")
print("Shape:", df.shape)

inspect_data(df)
X, y = separate_target(df)

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)
print("\nTarget distribution:")
print(y.value_counts())
X = remove_id_columns(X)

print("\nFeatures shape after removing IDs:", X.shape)
X = remove_high_missing_columns(X)

print("\nFeatures shape after removing high-missing columns:", X.shape)
categorical_columns = identify_categorical_columns(X)

print("\nNumber of categorical columns:", len(categorical_columns))
print("\nCategorical columns:")
print(categorical_columns)
print("\nCategorical columns and unique values:")

for col in categorical_columns:
    print(col, "→", X[col].nunique(), "unique values")
constant_columns = [
    col for col in X.columns
    if X[col].nunique() <= 1
]

print("\nConstant columns:")
print(constant_columns)
X = remove_constant_columns(X)

print("\nFeatures shape after removing constant columns:", X.shape)
X = encode_categorical_columns(X)

print("\nFeatures shape after encoding:", X.shape)
print("\nRemaining categorical columns:")
print(X.select_dtypes(include="object").columns.tolist())
X = extract_time_features(X)

print("\nFeatures shape after time feature engineering:", X.shape)

print("\nNew time features:")
print(X[["transaction_minute", "transaction_second"]].head())
X = extract_description_features(X)

print("\nFeatures shape after description feature engineering:", X.shape)

print("\nNew description features:")
print(X[["description_length", "description_word_count"]].head())
X = extract_location_features(X)

print("\nFeatures shape after location feature engineering:", X.shape)

print("\nNew location features:")
print(X[["latitude", "longitude"]].head())
X = extract_business_name_features(X)

print("\nFeatures shape after business name feature engineering:", X.shape)

print("\nNew business name feature:")
print(X["business_name_match_present"].value_counts())


X_train, X_test, y_train, y_test = split_data(X, y)

print("\nTraining features shape:", X_train.shape)
print("Testing features shape:", X_test.shape)

print("\nTraining target shape:", y_train.shape)
print("Testing target shape:", y_test.shape)

print("\nTraining fraud distribution:")
print(y_train.value_counts(normalize=True))

print("\nTesting fraud distribution:")
print(y_test.value_counts(normalize=True))

