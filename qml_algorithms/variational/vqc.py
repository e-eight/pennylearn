# This code is part of qml_algorithms.
#
# (C) Copyright Soham Pal 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

from typing import Any, Callable, Tuple, Union

import pennylane as qml
import pennylane.numpy as np
from numpy.typing import ArrayLike
from qml_algorithms.templates import Ansatz, Embedding


class VQC:
    r"""Variational Quantum Classifier.

    For the details of the algorithm check
    https://pennylane.ai/qml/demos/tutorial_variational_classifier.html.

    Args:
        embedding (Embedding): Embedding wrapping a 'non-trainable' template from
        PennyLane's template library.
        ansatz (Ansatz): Ansatz wrapping a 'trainable' template from PennyLane's
        template library.
        loss (Callable): The loss function that is to be minimized.
        optimizer (qml.optimize.GradientDescentOptimizer): Any optimizer included with
        PennyLane, or a custom optimizer that subclasses `GradientDescentOptimizer`.
        device (str): Name of device on which VQC will run.
    """

    def __init__(
        self,
        embedding: Embedding,
        ansatz: Ansatz,
        loss: Callable,
        optimizer: qml.optimize.GradientDescentOptimizer,
        device: str = "default.qubit",
    ):
        if embedding.num_wires != ansatz.num_wires:
            raise ValueError(
                "Embedding and ansatz must be applied to the same number of wires."
            )
        self._embedding = embedding
        self._ansatz = ansatz
        self._num_wires = self._embedding.num_wires
        self._loss = loss
        self._optimizer = optimizer
        self._device = qml.device(device, wires=self._num_wires)
        self._fit_result = None

    @property
    def embedding(self) -> Embedding:
        """Returns the embedding."""
        return self._embedding

    @property
    def ansatz(self) -> Ansatz:
        """Returns the ansatz."""
        return self._ansatz

    @property
    def num_wires(self) -> int:
        """Returns number of wires in the circuit."""
        return self._num_wires

    @property
    def loss(self) -> Callable:
        """Returns the loss function."""
        return self._loss

    @loss.setter
    def loss(self, loss):
        """Sets the loss."""
        self._loss = loss

    @property
    def optimizer(self) -> qml.optimize.GradientDescentOptimizer:
        """Returns the optimizer."""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: qml.optimize.GradientDescentOptimizer):
        """Sets the optimizer."""
        self._optimizer = optimizer

    @property
    def device(self) -> qml.Device:
        """Returns the device."""
        return self._device

    @device.setter
    def device(self, device: str):
        """Sets the device."""
        self._device = qml.device(device, wires=self._num_wires)

    def _circuit(self, weights: ArrayLike, x: ArrayLike) -> Callable:
        """The VQC circuit.

        Args:
            weights (ArrayLike): Array of weights for the ansatz.
            x (ArrayLike): n-dimensional datapoint.

        Returns:
            A function that can construct the circuit.
        """

        @qml.qnode(self._device)
        def circuit_helper():
            self._embedding.circuit(x)
            self._ansatz.circuit(weights)
            return qml.expval(qml.PauliZ(0))

        return circuit_helper()

    def _forward(self, var: Tuple[ArrayLike, float], x: ArrayLike) -> float:
        """Forward propagation of the VQC circuit.

        Args:
            var (Tuple[ArrayLike, float]): A tuple of an array, containing the weights
            for the circuit, and a float, which is the bias.
            x (ArrayLike): n-dimensional datapoint.

        Returns:
            The value after forward propagation of the datapoint.
        """
        weights = var[0]
        bias = var[1]
        return self._circuit(weights, x) + bias

    def fit(
        self,
        x: ArrayLike,
        y: ArrayLike,
        epochs: int = 40,
        batch_size: int = 5,
        seed: Union[int, ArrayLike, Any] = None,
        callback: Callable = None,
    ):
        """Fits the VQC to the training data, by optimizing the ansatz parameters.

        Args:
            x (ArrayLike): The input data, must have shape (n_samples, n_features).
            y (ArrayLike): The targets, must have shape (n_samples,).
            epochs (int, optional): Number of training iterations, default value is 40.
            batch_size (int, optional): Size of the training batch, default value is 5.
            seed (int, ArrayLike[int], SeedSequence, BitGenerator, Generator): Seed for
            numpy's random number generator.
            callback (Callable, optional): Callback function for status updates. Should
            have the signature (int, ArrayLike, float).
        """

        def objective(var, x_batch, y_batch):
            predictions = [self._forward(var, sample) for sample in x_batch]
            return self._loss(predictions, y_batch)

        rng = np.random.default_rng(seed=seed)
        var = (0.01 * rng.standard_normal(size=self._ansatz.shape), 0.0)
        for epoch in range(epochs):
            batch_index = rng.integers(0, len(x), (batch_size,))
            x_batch = x[batch_index]
            y_batch = y[batch_index]
            var = self._optimizer.step(lambda v: objective(v, x_batch, y_batch), var)

            if callback is not None:
                predictions = [np.sign(self._forward(var, sample)) for sample in x]
                cost = objective(var, x, y)
                callback(epoch, predictions, cost)

        self._fit_result = var

    def predict(self, x: ArrayLike):
        """Predict using the trained VQC model.

        Args:
            x (ArrayLike): The input data.

        Returns:
            The predicted classes.
        """
        if self._fit_result is None:
            raise ValueError("Model needs to be fitted to some training data")
        return [np.sign(self._forward(self._fit_result, sample) for sample in x)]
