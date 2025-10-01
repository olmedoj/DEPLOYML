from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
import numpy as np
from collections import Counter
import joblib

# Load the Pima Indians Diabetes dataset
print("Loading dataset...")
pima = fetch_openml('diabetes', as_frame=True)
X, y = pima.data, pima.target

# Print key information about the dataset
print(f"\nDataset shape: {X.shape}")
print(f"Features: {pima.feature_names}")
print(f"Target variable: {pima.target_names}")
print(f"Class distributions: {Counter(y)}")

# Encode target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Create pipeline with preprocessing and model
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        n_jobs=1,
        eval_metric='logloss'
    ))
])

# Define parameter grid for GridSearchCV
param_grid = {
    'classifier__max_depth': [3, 4, 5],
    'classifier__learning_rate': [0.01, 0.05, 0.1],
    'classifier__n_estimators': [50, 100, 200],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

# Perform grid search with cross-validation
print("\nPerforming GridSearchCV...")
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)

# Print best results
print(f"\n{'='*60}")
print(f"Best cross-validation score: {grid_search.best_score_:.4f}")
print(f"\nBest parameters:")
for param, value in grid_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"{'='*60}")

# Get best pipeline
best_pipeline = grid_search.best_estimator_

# Evaluate on test set
y_pred = best_pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTest Set Performance:")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Cross-validation scores on full training data
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Save the complete pipeline
pipeline_filename = 'diabetes_pipeline.pkl'
joblib.dump(best_pipeline, pipeline_filename)
print(f"\nPipeline saved to: {pipeline_filename}")

# Save the label encoder
joblib.dump(le, 'label_encoder.pkl')
print(f"Label encoder saved to: label_encoder.pkl")

# Demonstrate loading and using the saved pipeline
print("\n" + "="*60)
print("Testing saved pipeline...")
loaded_pipeline = joblib.load(pipeline_filename)
loaded_le = joblib.load('label_encoder.pkl')

# Make predictions with loaded pipeline
loaded_predictions = loaded_pipeline.predict(X_test)
loaded_accuracy = accuracy_score(y_test, loaded_predictions)
print(f"Loaded pipeline accuracy: {loaded_accuracy:.4f}")

# Example prediction on a single sample
sample = X_test.iloc[0:1]
prediction = loaded_pipeline.predict(sample)
prediction_proba = loaded_pipeline.predict_proba(sample)

print(f"\nExample prediction:")
print(f"Input features: {sample.values[0]}")
print(f"Predicted class: {loaded_le.inverse_transform(prediction)[0]}")
print(f"Prediction probabilities: {prediction_proba[0]}")
print(f"Actual class: {loaded_le.inverse_transform([y_test[0]])[0]}")
print("="*60)
