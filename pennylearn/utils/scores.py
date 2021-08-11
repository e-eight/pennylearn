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

""" Statistical scores """

import numpy as np
from numpy.typing import ArrayLike


def _validate_shapes(predict: ArrayLike, target: ArrayLike):
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


def accuracy(
    predict: ArrayLike, target: ArrayLike, rtol: float = 0.0, atol: float = 1e-5
) -> float:
    """
    Calculates the accuracy score.

    Args:
        predict: an array of predicted values using the model
        target: an array of the true values
        rtol: relative tolerance parameter for np.isclose
        atol: absolute tolerance parameter for np.isclose

    Returns:
        The accuracy score
    """
    predict_, target_ = np.asarray(predict), np.asarray(target)
    _validate_shapes(predict_, target_)
    return np.sum(np.isclose(predict_, target_, rtol=rtol, atol=atol)) / len(predict_)
