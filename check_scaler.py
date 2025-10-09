import joblib
scaler = joblib.load('scaler (1).pkl')
print("Number of features expected by scaler:", scaler.n_features_in_)