# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com
"""Typing support."""

from typing import (
    Callable,
)
from numpy.typing import (
    NDArray,
)
from .PETSc import (
    Vec,
    Mat,
    SNES,
    TS,
    TAO,
)

__all__ = [
    "Scalar",
    "TSRHSFunction",
    "TSRHSJacobian",
    "TSRHSJacobianP",
    "TSIFunction",
    "TSIJacobian",
    "TSIJacobianP",
    "TSI2Function",
    "TSI2Jacobian",
    "TSI2JacobianP",
    "TSMonitorFunction",
    "TSPreStepFunction",
    "TSPostStepFunction",
    "TSEventHandlerFunction",
    "TSPostEventFunction",
    "TSPreStepFunction",
    "TSPostStepFunction",
]


Scalar = float | complex
"""Scalar type.

Scalars can be either `float` or `complex` (but not both) depending on how
PETSc was configured (``./congigure --with-scalar-type=real|complex``).

"""

# --- TS ---

TSRHSFunction = Callable[[TS, float, Vec, Vec], None]
"""`TS` right hand side function callback."""

TSRHSJacobian = Callable[[TS, float, Vec, Mat, Mat], None]
"""`TS` right hand side Jacobian callback."""

TSRHSJacobianP = Callable[[TS, float, Vec, Mat], None]
"""`TS` right hand side parameter Jacobian callback."""

TSIFunction = Callable[[TS, float, Vec, Vec, Vec], None]
"""`TS` implicit function callback."""

TSIJacobian = Callable[[TS, float, Vec, Vec, float, Mat, Mat], None]
"""`TS` implicit Jacobian callback."""

TSIJacobianP = Callable[[TS, float, Vec, Vec, float, Mat], None]
"""`TS` implicit parameter Jacobian callback."""

TSI2Function = Callable[[TS, float, Vec, Vec, Vec, Vec], None]
"""`TS` implicit 2nd order function callback."""

TSI2Jacobian = Callable[[TS, float, Vec, Vec, Vec, float, float, Mat, Mat], None]
"""`TS` implicit 2nd order Jacobian callback."""

TSI2JacobianP = Callable[[TS, float, Vec, Vec, Vec, float, float, Mat], None]
"""`TS` implicit 2nd order parameter Jacobian callback."""

TSMonitorFunction = Callable[[TS, int, float, Vec], None]
"""`TS` monitor function."""

TSPreStepFunction = Callable[[TS], None]
"""`TS` pre-step function."""

TSPostStepFunction = Callable[[TS], None]
"""`TS` post-step function."""

TSEventHandlerFunction = Callable[[TS, float, Vec, NDArray[Scalar]], None]
"""`TS` event handler function."""

TSPostEventFunction = Callable[[TS, NDArray[int], float, Vec, bool], None]
"""`TS` post-event handler function."""

# --- TAO ---
