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

from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pennylane as qml

# import pennylane.numpy as pnp
from numpy.typing import ArrayLike
from pennylearn.templates import Ansatz, Embedding
from pennylearn.utils.scores import accuracy


class VQE:
    """Variational Quantum Eigensolver for finding the ground state energy of quantum
    systems."""

    def __init__(
        self,
        initial_state: ArrayLike,
        ansatz: Ansatz,
        hamiltonian: qml.Hamiltonian,
        optimizer: qml.optimize.GradientDescentOptimizer,
        device: str = "default.qubit",
        **kwargs,
    ):
        max_num_wires = max(
            [
                len(op.obs)
                for op in hamiltonian.ops
                if isinstance(op, qml.operation.Tensor)
            ]
        )
        if ansatz.num_wires < max_num_wires:
            raise ValueError(
                "Ansatz is not applied to as many wires as required for the Hamiltonian,"
                + f"{ansatz.num_wires} != {max_num_wires}."
            )
        if len(initial_state) != ansatz.num_wires:
            raise ValueError(
                "Initial state and ansatz must be applied to the same number of wires,"
                + f"{len(initial_state)} != {ansatz.num_wires}."
            )
        self._initial_state = initial_state
        self._ansatz = ansatz
        self._hamiltonian = hamiltonian
        self._num_wires = self._ansatz.num_wires
        self._optimizer = optimizer
        self._device = qml.device(device, wires=self._num_wires, **kwargs)
        self._ground_state_energy = None

    @property
    def initial_state(self) -> ArrayLike:
        """Returns the initial state."""
        return self._initial_state

    @property
    def ansatz(self) -> Ansatz:
        """Returns the ansatz."""
        return self._ansatz

    @property
    def num_wires(self) -> int:
        """Returns the number of wires in the circuit."""
        return self._num_wires

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
    def device(self, device: str, **kwargs):
        """Sets the device."""
        self._device = qml.device(device, wires=self._num_wires, **kwargs)

    def _circuit(self, params: ArrayLike, wires: ArrayLike):
        """The VQE circuit."""
        qml.BasisState(self._initial_state, wires=wires)
        self._ansatz.circuit(params)

    def fit(
        self,
        epochs: int = 40,
        tol: float = 1e-6,
        optimize: bool = False,
        seed: Union[int, ArrayLike, Any] = None,
        callback: Callable = None,
    ):
        """Optimizes the ansatz parameters by minimizing the cost function."""

        objective = qml.ExpvalCost(
            self._circuit, self._hamiltonian, self._device, optimize=optimize
        )

        rng = np.random.default_rng(seed=seed)
        params = 0.01 * rng.standard_normal(size=self._ansatz.shape)
        for epoch in range(epochs):
            params, prev_cost = self._optimizer.step_and_cost(objective, params)
            curr_cost = objective(params)

            if callback is not None:
                callback(epoch, curr_cost)

            if np.abs(curr_cost - prev_cost) < tol:
                break

        self._ground_state_energy = objective(params)

    def predict(self):
        """Returns the computed ground state energy."""
        if self._ground_state_energy is None:
            raise ValueError("The ansatz needs to be optimized.")
        return self._ground_state_energy
