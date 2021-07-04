from qml_algorithms.kernels import QuantumKernel
from sklearn.svm import SVC, SVR


class QSVC(SVC):
    """Quantum Support Vector Classifier.

    This extends the SVC class from `scikit-learn` to accept quantum kernels.

    Args:
        *args: Arbitrary arguments to pass to the SVC generator.
        quantum_kernel (QuantumKernel): A quantum kernel, check `qml_algorithms.kernels`
        for details.
        **kwargs: Arbitrary keyword arguments to pass to the SVC generator.
    """

    def __init__(self, *args, quantum_kernel: QuantumKernel, **kwargs):
        self._quantum_kernel = quantum_kernel
        super().__init__(kernel=self._quantum_kernel.evaluate, *args, **kwargs)

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns the quantum kernel."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel):
        """Sets the quantum kernel."""
        self._quantum_kernel = quantum_kernel
        self.kernel = self._quantum_kernel.evaluate


class QSVR(SVR):
    """Quantum Support Vector Regressor.

    This extends the SVR class from `scikit-learn` to accept quantum kernels.

    Args:
        *args: Arbitrary arguments to pass to the SVR generator.
        quantum_kernel (QuantumKernel): A quantum kernel, check `qml_algorithms.kernels`
        for details.
        **kwargs: Arbitrary keyword arguments to pass to the SVR generator.
    """

    def __init__(self, *args, quantum_kernel: QuantumKernel, **kwargs):
        self._quantum_kernel = quantum_kernel
        super().__init__(kernel=self._quantum_kernel.evaluate, *args, **kwargs)

    @property
    def quantum_kernel(self) -> QuantumKernel:
        """Returns the quantum kernel."""
        return self._quantum_kernel

    @quantum_kernel.setter
    def quantum_kernel(self, quantum_kernel: QuantumKernel):
        """Sets the quantum kernel."""
        self._quantum_kernel = quantum_kernel
        self.kernel = self._quantum_kernel.evaluate
