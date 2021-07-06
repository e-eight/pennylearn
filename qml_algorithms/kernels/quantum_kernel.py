# Copyright 2021 Soham Pal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pennylane as qml
from numpy.typing import ArrayLike
from pennylane.kernels import kernel_matrix
from qml_algorithms.templates import Embedding


class QuantumKernel:
    r"""Quantum Kernel.

    The quantum kernel algorithm calculates a kernel matrix, given datapoints x and y
    and embedding f, all of dimension n, to be used with classical kernel learning
    techniques like support vector machines, and ridge regression.

    The kernel function is :math:`k(x, y) = \langle f(x), f(y) \rangle`. The embedding
    maps a datapoint from n-dimensional space to m-dimensional space. Usually m is much
    larger than n.

    Args:
        embedding (Embedding): Embedding wrapping a 'non-trainable' template from
        PennyLane's template library.
        device (str, optional): Name of the device (simulator) on which the quantum
        circuit will run. The default value is `default.qubit`.
    """

    def __init__(
        self,
        embedding: Embedding,
        device: str = "default.qubit",
    ):
        self._embedding = embedding
        self._num_wires = self._embedding.num_wires
        self._device = qml.device(device, wires=self._num_wires)

    @property
    def embedding(self) -> Embedding:
        """Returns the embedding."""
        return self._embedding

    @embedding.setter
    def embedding(self, embedding: Embedding):
        """Sets the embedding."""
        self._embedding = embedding

    @property
    def device(self) -> qml.Device:
        """Returns the device."""
        return self._device

    @device.setter
    def device(self, device: str):
        """Sets the device."""
        self._device = qml.device(device, wires=self._num_wires)

    def kernel(self, x: float, y: float) -> float:
        """Kernel function, :math:`k(x, y) = \langle f(x), f(y) \rangle`.

        Args:
            x, y (float): n-dimensional datapoints.
        Returns:
            The corresponding entry of the kernel matrix.
        """
        projector = np.zeros((2 ** self._num_wires, 2 ** self._num_wires))
        projector[0, 0] = 1
        embedding = self._embedding.circuit()

        @qml.qnode(self._device)
        def kernel_helper():
            embedding(x, wires=range(self._num_wires))
            qml.adjoint(embedding)(y, wires=range(self._num_wires))
            return qml.expval(qml.Hermitian(projector, wires=range(self._num_wires)))

        return kernel_helper()

    def evaluate(self, x: ArrayLike, y: ArrayLike) -> ArrayLike:
        """Calculates the kernel matrix.

        Args:
            x, y (ArrayLike): Data arrays.
        Returns:
            The kernel matrix.
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if x.ndim > 2:
            raise ValueError("x must be a 1D or 2D array.")
        if x.ndim == 1:
            x = np.reshape(x, (-1, 2))

        if not isinstance(y, np.ndarray):
            y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y must be a 1D or 2D array.")
        if y.ndim == 1:
            y = np.reshape(y, (-1, 2))

        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"incompatible dimensions, x: {x.shape[1]} != y: {y.shape[1]}"
            )

        return kernel_matrix(x, y, self.kernel)
