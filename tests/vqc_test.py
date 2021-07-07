#!/bin/bash/env python

# PennyLane imports
import pennylane as qml
import pennylane.numpy as np
from pennylane.optimize import AdamOptimizer
from pennylane.templates.layers import StronglyEntanglingLayers

# PennyLearn imports
from pennylearn.templates import Ansatz, Embedding
from pennylearn.variational import VQC

# scikit-learn imports
from sklearn.metrics import accuracy_score

# Parity

print("Parity Test")
print("===========\n")

data = np.loadtxt("./parity.txt")
X = np.array(data[:, :-1], requires_grad=False)
Y = np.array(data[:, -1], requires_grad=False)
Y = Y * 2 - np.ones(len(Y))

print("Data:")
print("-----")
for i in range(5):
    print("X = {}, Y = {: d}".format(X[i], int(Y[i])))

print("...\n")

num_wires = X.shape[1]
embedding = Embedding(template=qml.BasisState, num_wires=num_wires)
ansatz = Ansatz(template=StronglyEntanglingLayers, num_layers=2, num_wires=num_wires)


def square_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p) ** 2
    loss = loss / len(labels)
    return loss


device = "default.qubit"

vqc = VQC(
    embedding,
    ansatz,
    square_loss,
    AdamOptimizer(stepsize=0.5),
    device,
)


def callback(epoch, predictions, cost):
    accuracy = accuracy_score(predictions, Y)
    print(f"Iteration: {epoch + 1:5d} | Cost: {cost:0.7f} | Accuracy: {accuracy:0.7f}")


print("Traning:")
print("--------")
vqc.fit(X, Y, seed=0, epochs=25, callback=callback)
