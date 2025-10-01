from sklearn.datasets import fetch_openml
import numpy as np


# Load the Pima Indians Diabetes dataset
pima = fetch_openml('diabetes', as_frame=True)
print(pima)
X, y = pima.data, pima.target
