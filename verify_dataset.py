import pandas as pd

# Load dataset
data = pd.read_csv('pd_speech_features.csv')

# Print info
print("Dataset shape (rows, columns):", data.shape)
print("\nFirst 10 columns:", data.columns[:10].tolist())
print("Last 5 columns:", data.columns[-5:].tolist())
print("\nNumber of features (excluding id, class):", len(data.columns) - 2)
print("\nClass distribution:")
print(data['class'].value_counts())