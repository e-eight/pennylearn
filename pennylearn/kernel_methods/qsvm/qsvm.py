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


from sklearn.svm import SVC, SVR

from ..base import BaseKernelMethod
from ..kernels import QuantumKernel


class QSVC(BaseKernelMethod):
    """Quantum Support Vector Classifier.

    This class extends the SVC class from `sklearn.svm` to accept quantum kernels.

    Args:
        quantum_kernel (QuantumKernel): A quantum kernel. Check `pennylane.kernels` for
        details.
        **kwargs: Arbitrary keyword arguments to be passed to the SVC constructor.
    """

    def __init__(self, quantum_kernel: QuantumKernel, **kwargs):
        super().__init__(SVC, quantum_kernel, **kwargs)


class QSVR(BaseKernelMethod):
    """Quantum Support Vector Regressor.

    This class extends the SVR class from `sklearn.svm` to accept quantum kernels.

    Args:
        quantum_kernel (QuantumKernel): A quantum kernel. Check `pennylane.kernels` for
        details.
        **kwargs: Arbitrary keyword arguments to be passed to the SVR constructor.
    """

    def __init__(self, quantum_kernel: QuantumKernel, **kwargs):
        super().__init__(SVR, quantum_kernel, **kwargs)
