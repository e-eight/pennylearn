#!/bin/bash/env python
import numpy as np
import pennylane as qml
from pennylearn.templates import Ansatz
from pennylearn.variational import VQE

# Hamiltonian
h_matrix = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, -1, 0, 0], [0, 0, 0, 1]])
h_operator = qml.utils.decompose_hamiltonian(h_matrix)
h_operator = qml.Hamiltonian(*h_operator, simplify=True)
print(h_operator)

# ansatz
ansatz = Ansatz(qml.templates.StronglyEntanglingLayers, 2, 2)

# optimizer
optimizer = qml.optimize.AdamOptimizer()

# exact eigenvalues
eigvals = np.linalg.eigvals(h_matrix)
print(eigvals)

# vqe
def callback(epoch, cost):
    if epoch % 2 == 0:
        print(f"Epoch: {epoch + 1}, Cost: {cost}")


vqe = VQE(np.array([0, 0]), ansatz, h_operator, optimizer)
vqe.fit(epochs=300, callback=callback)
print(vqe.predict())
