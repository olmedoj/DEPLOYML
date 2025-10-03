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

# Encode target variable
print(y.value_counts())
y = LabelEncoder().fit_transform(y)
print(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define parameter grid
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}

# Create XGBClassifier
model = XGBClassifier(objective='binary:logistic', random_state=42, n_jobs=1)

# Perform grid search
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)

# Print best score and parameters
print(f"Best score: {grid_search.best_score_:.3f}")
print(f"Best parameters: {grid_search.best_params_}")

# Access best model
best_model = grid_search.best_estimator_

# Save best model
best_model.save_model('best_model_pima.ubj')

# Load saved model
loaded_model = XGBClassifier()
loaded_model.load_model('best_model_pima.ubj')

# Use loaded model for predictions
predictions = loaded_model.predict(X_test)

# Print accuracy score
accuracy = loaded_model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.3f}")