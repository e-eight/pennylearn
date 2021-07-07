#!/bin/bash/env python

import numpy as np

# PennyLane imports
from pennylane.templates import AmplitudeEmbedding, AngleEmbedding

# PennyLane imports
from pennylearn.kernels import QuantumKernel
from pennylearn.qsvm import QSVC
from pennylearn.templates import Embedding

# scikit-learn imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# scikit-learn example
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

print("scikit-learn example")
print("====================\n")

X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
y = np.array([1, 1, 2, 2])

print("Data:")
print("-----")
for i in range(4):
    print("X = {}, y = {: d}".format(X[i], int(y[i])))
print("\n")

embedding = Embedding(
    template=AmplitudeEmbedding, num_wires=2, pad_with=True, normalize=True
)
qkernel = QuantumKernel(embedding, "default.qubit")
qsvc = QSVC(quantum_kernel=qkernel)
print("Fitting...")
qsvc.fit(X, y)
print(f"Prediction at X = [[-0.8, -1]]: {qsvc.predict([[-0.8, -1]])}")

print("\n")

# PennyLane example
# https://pennylane.ai/qml/demos/tutorial_kernel_based_training.html

print("PennyLane example")
print("=================\n")

X, y = load_iris(return_X_y=True)
X = X[:100]
y = y[:100]

print("Data:")
print("-----")
for i in range(5):
    print("X = {}, y = {: d}".format(X[i], int(y[i])))
print("...\n")

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

y_scaled = 2 * (y - 0.5)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled)

embedding = Embedding(template=AngleEmbedding, num_wires=X_train.shape[1])
qkernel = QuantumKernel(embedding, "default.qubit")
qsvc = QSVC(quantum_kernel=qkernel)
print("Fitting...")
qsvc.fit(X_train, y_train)
print(f"Accuracy score after fitting: {qsvc.score(X_test, y_test)}")
