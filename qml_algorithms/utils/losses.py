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

""" Loss functions """

import numpy as np


def _validate_shapes(predict: np.ndarray, target: np.ndarray):
    """
    Validates that shapes of both parameters are identical.

    Args:
        predict: an array of predicted values using the model
        target: an array of the true values

    Raises:
        ValueError: shapes of predict and target do not match
    """

    if predict.shape != target.shape:
        raise ValueError(
            f"shape mismatch, predict: {predict.shape} != target: {target.shape}"
        )


def cross_entropy_loss(predict: np.ndarray, target: np.ndarray):
    r"""
    Calculates the cross entropy loss for each sample:

    .. math::

        \text{cross_entropy_loss}(predict, target) =
        -\sum_{i=0}^{N_{\text{classes}}} target_i * log(predict_i)

    Args:
        predict: an array of predicted values using the model
        target: an array of the true values

    Returns:
        an array with the cross entropy loss for each sample
    """
    _validate_shapes(predict, target)
    if len(predict.shape == 1):
        predict = predict[:, None]
        target = target[:, None]

    loss = -np.einsum("ij,ij->i", target, np.log2(predict)).reshape(-1, 1)
    return loss


def binary_cross_entropy_loss(predict: np.ndarray, target: np.ndarray):
    r"""
    Calculates the cross entropy loss for each sample for binary classification.

    Args:
        predict: an array of predicted values using the model
        target: an array of the true values

    Returns:
        an array with the binary cross entropy loss for each sample
    """
    _validate_shapes(predict, target)
    if len(set(target) != 2):
        raise ValueError(
            "Binary cross entropy loss should only be used for \
        binary classification"
        )

    return 1.0 / (1.0 + np.exp(-cross_entropy_loss(predict, target)))
