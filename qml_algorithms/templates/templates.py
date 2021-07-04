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

from functools import partial

import pennylane as qml
from numpy.typing import ArrayLike


class Embedding:
    r"""Embedding classical data in to quantum states.

    Embeddings, and layers defined in the templates library of PennyLane need the actual
    data array (features or weights) to be constructed. This class wraps an embedding
    from the templates library so that it can be passed to caller functions without
    being actually constructed until necessary. Technically this can also be used to
    wrap layers, but it is better to use the `Ansatz` subclass defined below.

    Args:
        template (qml.operation.Operation): An embedding from the templates library of
        PennyLane. Though `QAOAEmbedding` can be used with this class, it is better to
        use it with the `Ansatz` subclass defined below.
        num_wires (int): The number of wires on which the embedding will act on.
        **kwargs: Arbitrary keyword arguments to pass to the template constructor.
    """

    def __init__(self, template: qml.operation.Operation, num_wires: int, **kwargs):
        self._template = template
        self._num_wires = num_wires
        self._circuit = partial(self._template, wires=range(self._num_wires), **kwargs)

    @property
    def template(self) -> qml.operation.Operation:
        """Returns the embedding template."""
        return self._template

    @template.setter
    def template(self, template: qml.operation.Operation):
        """Sets the embedding template."""
        self._template = template

    @property
    def num_wires(self) -> int:
        """Returns the number of wires."""
        return self._num_wires

    @num_wires.setter
    def num_wires(self, num_wires: int):
        """Sets the number of wires."""
        self._num_wires = num_wires

    def circuit(self, features_or_weights: ArrayLike):
        """Constructs the embedding template and returns it."""
        return self._circuit(features_or_weights)


class Ansatz(Embedding):
    def __init__(
        self,
        template: qml.operation.Operation,
        num_layers: int,
        num_wires: int,
        **kwargs
    ):
        r"""Ansatz for variational algorithms.

        This class wraps a trainable layer from the templates library of PennyLane so
        that it can be passed to caller functions without being actually constructed
        until necessary.

        Args:
            template (qml.operation.Operation): A layer from the templates library of
            PennyLane.
            num_layers (int): Number of layers in the ansatz.
            num_wires (int): The number of wires on which the embedding will act on.
            **kwargs: Arbitrary keyword arguments to pass to the template constructor.
        """
        super().__init__(template, num_wires, **kwargs)
        self._num_layers = num_layers
        self._shape = self._template.shape(self._num_layers, self._num_wires)

    @property
    def num_layers(self) -> int:
        """Returns the number of layers."""
        return self._num_layers

    @num_layers.setter
    def num_layers(self, num_layers: int):
        """Sets the number of layers."""
        self._num_layers = num_layers

    @property
    def shape(self) -> ArrayLike:
        """Returns the shape of the weight tensor required for the template."""
        return self._shape
