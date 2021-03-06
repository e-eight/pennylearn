PennyLearn provides a scikit-learn like API for
[PennyLane](https://github.com/PennyLaneAI/pennylane) to easily
prototype quantum machine learning ideas.

# Installation

PennyLearn requires Python 3.7 and up. PennyLearn and its dependencies
can be installed using pip:

``` bash
pip install git+https://github.com/e-eight/pennylearn
```

# Features

PennyLearn is still in alpha stage, so things are likely to change.
Currently the following algorithms have been implemented:

-   Quantum Support Vector Classifier
-   Quantum Support Vector Regressor
-   Variational Quantum Classifier

These have been tested with the default PennyLane simulator, and
interface. Further testing is required to see if they would work with
the alternate backends and interfaces supported by PennyLane. Currently
I plan to add all the algorithms implemented in [Qiskit Machine
Learning](https://github.com/Qiskit/qiskit-machine-learning).

# Usage

## Quantum Support Vector Classifier

In this example we use angle embedding to map a subset of the Iris
dataset to quantum states. The quantum support vector classifier is then
trained on this embedded dataset. For this we need to first wrap
`AngleEmbedding` class from PennyLane with the `Embedding` class from
PennyLearn so that it can be used to build a quantum kernel usable by
the quantum support vector classifier.

``` python
import numpy as np

# PennyLane imports
from pennylane.templates import AmplitudeEmbedding, AngleEmbedding

# PennyLane imports
from pennylearn.kernel_methods.kernels import QuantumKernel
from pennylearn.kernel_methods.qsvm import QSVC
from pennylearn.templates import Embedding

# scikit-learn imports
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
qsvc.fit(X_train, y_train)
print(f"Accuracy score after fitting: {qsvc.score(X_test, y_test)}")
```

## Variational Quantum Classifier

In this example we recreate the "parity" example from the PennyLane
tutorials. The data is mapped to quantum states using the `BasisState`
embedding from PennyLane, and the ansatz is `StronglyEntanglingLayers`
from PennyLane. The classifier is optimized using PennyLane's
`AdamOptimizer`.

``` python
# PennyLane imports
import pennylane as qml
import pennylane.numpy as np
from pennylane.optimize import AdamOptimizer
from pennylane.templates.layers import StronglyEntanglingLayers

# PennyLearn imports
from pennylearn.templates import Ansatz, Embedding
from pennylearn.utils.scores import accuracy
from pennylearn.variational import VQC

data = np.loadtxt("./parity.txt")
X = np.array(data[:, :-1], requires_grad=False)
Y = np.array(data[:, -1], requires_grad=False)
Y = Y * 2 - np.ones(len(Y))

print("Data:")
print("-----")
for i in range(5):
    print("X = {}, Y = {: d}".format(X[i], int(Y[i])))
print(X.shape)
print(Y.shape)

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
    acc = accuracy(predictions, Y)
    print(f"Iteration: {epoch + 1:5d} | Cost: {cost:0.7f} | Accuracy: {acc:0.7f}")


print("Traning:")
print("--------")
vqc.fit(X, Y, seed=0, epochs=25, callback=callback)
print(f"Final score: {vqc.score(X, Y)}")  
```

# License

PennyLearn is free and open source, released under the Apache License,
Version 2.0.
