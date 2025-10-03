from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV, train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter
from sklearn.pipeline import Pipeline
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the Pima Indians Diabetes dataset
pima = fetch_openml('diabetes', as_frame=True)
X, y = pima.data, pima.target

label_encoder = LabelEncoder().fit(y)
y = label_encoder.transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", XGBClassifier( objective='binary:logistic',
        random_state=42,
        n_jobs=1,
        eval_metric="logloss"))
])


param_grid = {
    "classifier__max_depth": [3, 4, 5],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__n_estimators": [50, 100, 200],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)

print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
print("Best estimator:", grid_search.best_estimator_)

y_pred = grid_search.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))

# save pipeline
pipeline_name = "modelo_diabetes.pkl"
joblib.dump(grid_search.best_estimator_, pipeline_name)
# save labelencoder
label_encoder_name = "label_encoder_diabetes.pkl"
joblib.dump(label_encoder, label_encoder_name)

