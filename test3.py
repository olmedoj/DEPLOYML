from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter

# Load the Pima Indians Diabetes dataset
pima = fetch_openml('diabetes', as_frame=True)
X, y = pima.data, pima.target

# Print key information about the dataset
print(f"Dataset shape: {X.shape}")
print(f"Features: {pima.feature_names}")
print(f"Target variable: {pima.target_names}")
print(f"Class distributions: {Counter(y)}")