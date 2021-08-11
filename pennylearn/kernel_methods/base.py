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

from typing import Any, Optional

from numpy.typing import ArrayLike

from .kernels import QuantumKernel


class BaseKernelMethod:
    """Base method for all kernel methods.

    This extends kernel methods in scikit-learn to accept quantum kernels.

    Args:
        model: Kernel method from scikit-learn.
        quantum_kernel (QuantumKernel): A quantum kernel, check `quantum_kernel.py``
        for details.
        **kwargs: Arbitrary keyword arguments to pass to the kernel method.
    """

    def __init__(self, model: Any, quantum_kernel: QuantumKernel = None, **kwargs):
        if quantum_kernel is None:
            raise ValueError("`quantum_kernel` must be provided.")
        self._quantum_kernel = quantum_kernel
        self._model = model(kernel=self._quantum_kernel.evaluate, **kwargs)

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns the quantum kernel."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel):
        """Sets the quantum kernel."""
        self._quantum_kernel = quantum_kernel
        self._model.kernel = self._quantum_kernel.evaluate

    def fit(
        self, x: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ):
        """Fits the QSVC to the data."""
        self._model.fit(x, y, sample_weight)

    def predict(self, x: ArrayLike) -> ArrayLike:
        """Predicts using the fitted QSVC."""
        return self._model.predict(x)

    def score(
        self, x: ArrayLike, y: ArrayLike, sample_weight: Optional[ArrayLike] = None
    ) -> float:
        """Returns the mean accuracy on the given test data and labels."""
        return self._model.score(x, y, sample_weight)
