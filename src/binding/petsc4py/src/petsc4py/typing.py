# Author:  Lisandro Dalcin
# Contact: dalcinl@gmail.com
"""Typing support."""

from __future__ import annotations
from typing import (
    Callable,
    Sequence,
    Literal,
)
from numpy.typing import (
    NDArray,
)
from .PETSc import (
    InsertMode,
    ScatterMode,
    NormType,
    Vec,
    Mat,
    NullSpace,
    KSP,
    SNES,
    TS,
    TAO,
    DM,
)

__all__ = [
    "Scalar",
    "ArrayInt",
    "ArrayReal",
    "ArrayComplex",
    "ArrayScalar",
    "DimsSpec",
    "AccessModeSpec",
    "InsertModeSpec",
    "ScatterModeSpec",
    "LayoutSizeSpec",
    "NormTypeSpec",
    "MatAssemblySpec",
    "MatSizeSpec",
    "MatBlockSizeSpec",
    "CSRIndicesSpec",
    "CSRSpec",
    "NNZSpec",
    "MatNullFunction",
    "DMCoarsenHookFunction",
    "DMRestrictHookFunction",
    "KSPRHSFunction",
    "KSPOperatorsFunction",
    "KSPConvergenceTestFunction",
    "KSPMonitorFunction",
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
    "TAOObjectiveFunction",
    "TAOGradientFunction",
    "TAOObjectiveGradientFunction",
    "TAOHessianFunction",
    "TAOUpdateFunction",
    "TAOMonitorFunction",
    "TAOConvergedFunction",
    "TAOJacobianFunction",
    "TAOResidualFunction",
    "TAOJacobianResidualFunction",
    "TAOVariableBoundsFunction",
    "TAOConstraintsFunction",
]

# --- Sys ---

Scalar = float | complex
"""Scalar type.

Scalars can be either `float` or `complex` (but not both) depending on how
PETSc was configured (``./configure --with-scalar-type=real|complex``).

"""

ArrayInt = NDArray[int]
"""Array of `int`."""

ArrayReal = NDArray[float]
"""Array of `float`."""

ArrayComplex = NDArray[complex]
"""Array of `complex`."""

ArrayScalar = NDArray[Scalar]
"""Array of `Scalar` numbers."""

DimsSpec = tuple[int, ...]
"""Dimensions specification.

   N-tuples indicates N-dimensional grid sizes.

"""

AccessModeSpec = Literal['rw', 'r', 'w'] | None
"""Access mode specification.

   Possible values are:
     - ``'rw'`` Read-Write mode.
     - ``'r'`` Read-only mode.
     - ``'w'`` Write-only mode.
     - `None` as ``'rw'``.

"""

InsertModeSpec = InsertMode | bool | None
"""Insertion mode specification.

   Possible values are:
     - `InsertMode.ADD_VALUES` Add new value to existing one.
     - `InsertMode.INSERT_VALUES` Replace existing entry with new value.
     - `None` as `InsertMode.INSERT_VALUES`.
     - `False` as `InsertMode.INSERT_VALUES`.
     - `True` as `InsertMode.ADD_VALUES`.

   See Also
   --------
   InsertMode

"""

ScatterModeSpec = ScatterMode | bool | str | None
"""Scatter mode specification.

   Possible values are:
     - `ScatterMode.FORWARD` Forward mode.
     - `ScatterMode.REVERSE` Reverse mode.
     - `None` as `ScatterMode.FORWARD`.
     - `False` as `ScatterMode.FORWARD`.
     - `True` as `ScatterMode.REVERSE`.
     - ``'forward'`` as `ScatterMode.FORWARD`.
     - ``'reverse'`` as `ScatterMode.REVERSE`.

   See Also
   --------
   ScatterMode

"""

LayoutSizeSpec = int | tuple[int, int]
"""`int` or 2-`tuple` of `int` describing the layout sizes.

   A single `int` indicates global size.
   A `tuple` of `int` indicates ``(local_size, global_size)``.

   See Also
   --------
   Sys.splitOwnership

"""

NormTypeSpec = NormType | None
"""Norm type specification.

    Possible values include:

    - `NormType.NORM_1` The 1-norm: Σₙ abs(xₙ) for vectors, maxₙ (Σᵢ abs(xₙᵢ)) for matrices.
    - `NormType.NORM_2` The 2-norm: √(Σₙ xₙ²) for vectors, largest singular values for matrices.
    - `NormType.NORM_INFINITY` The ∞-norm: maxₙ abs(xₙ) for vectors, maxᵢ (Σₙ abs(xₙᵢ)) for matrices.
    - `NormType.NORM_FROBENIUS` The Frobenius norm: same as 2-norm for vectors, √(Σₙᵢ xₙᵢ²) for matrices.
    - `NormType.NORM_1_AND_2` Compute both `NormType.NORM_1` and `NormType.NORM_2`.
    - `None` as `NormType.NORM_2` for vectors, `NormType.NORM_FROBENIUS` for matrices.

    See Also
    --------
    PETSc.NormType, petsc.NormType

"""

# --- Mat ---

MatAssemblySpec = Mat.AssemblyType | bool | None
"""Matrix assembly specification.

   Possible values are:
     - `Mat.AssemblyType.FINAL`
     - `Mat.AssemblyType.FLUSH`
     - `None` as `Mat.AssemblyType.FINAL`
     - `False` as `Mat.AssemblyType.FINAL`
     - `True` as `Mat.AssemblyType.FLUSH`

   See Also
   --------
   petsc.MatAssemblyType

"""

MatSizeSpec = int | tuple[int, int] | tuple[tuple[int, int], tuple[int, int]]
"""`int` or (nested) `tuple` of `int` describing the matrix sizes.

   If `int` then rows = columns.
   A single `tuple` of `int` indicates ``(rows, columns)``.
   A nested `tuple` of `int` indicates ``((local_rows, rows), (local_columns, columns))``.

   See Also
   --------
   Sys.splitOwnership

"""

MatBlockSizeSpec = int | tuple[int, int]
"""The row and column block sizes.

   If a single `int` is provided then rows and columns share the same block size.

"""

CSRIndicesSpec = tuple[Sequence[int], Sequence[int]]
"""CSR indices format specification.

   A 2-tuple carrying the ``(row_start, col_indices)`` information.

"""

CSRSpec = tuple[Sequence[int], Sequence[int], Sequence[Scalar]]
"""CSR format specification.

   A 3-tuple carrying the ``(row_start, col_indices, values)`` information.

"""

NNZSpec = int | Sequence[int] | tuple[Sequence[int], Sequence[int]]
"""Nonzero pattern specification.

   A single `int` corresponds to fixed number of non-zeros per row.
   A `Sequence` of `int` indicates different non-zeros per row.
   If a 2-`tuple` is used, the elements of the tuple corresponds
   to the on-process and off-process parts of the matrix.

   See Also
   --------
   petsc.MatSeqAIJSetPreallocation, petsc.MatMPIAIJSetPreallocation

"""

# --- MatNullSpace ---

MatNullFunction = Callable[[NullSpace, Vec], None]
"""`PETSc.NullSpace` callback."""

# --- DM ---

DMCoarsenHookFunction = Callable[[DM, DM], None]
"""`PETSc.DM` coarsening hook callback."""

DMRestrictHookFunction = Callable[[DM, Mat, Vec, Mat, DM], None]
"""`PETSc.DM` restriction hook callback."""

# --- KSP ---

KSPRHSFunction = Callable[[KSP, Vec], None]
"""`PETSc.KSP` right hand side function callback."""

KSPOperatorsFunction = Callable[[KSP, Mat, Mat], None]
"""`PETSc.KSP` operators function callback."""

KSPConvergenceTestFunction = Callable[[KSP, int, float], KSP.ConvergedReason]
"""`PETSc.KSP` convergence test callback."""

KSPMonitorFunction = Callable[[KSP, int, float], None]
"""`PETSc.KSP` monitor callback."""

# --- SNES ---

SNESMonitorFunction = Callable[[SNES, int, float], None]
"""`SNES` monitor callback."""

SNESObjFunction = Callable[[SNES, Vec], None]
"""`SNES` objective function callback."""

SNESFunction = Callable[[SNES, Vec, Vec], None]
"""`SNES` residual function callback."""

SNESJacobianFunction = Callable[[SNES, Vec, Mat, Mat], None]
"""`SNES` Jacobian callback."""

SNESGuessFunction = Callable[[SNES, Vec], None]
"""`SNES` initial guess callback."""

SNESUpdateFunction = Callable[[SNES, int], None]
"""`SNES` step update callback."""

SNESLSPreFunction = Callable[[Vec, Vec], None]
"""`SNES` linesearch pre-check update callback."""

SNESNGSFunction = Callable[[SNES, Vec, Vec], None]
"""`SNES` nonlinear Gauss-Seidel callback."""

SNESConvergedFunction = Callable[[SNES, int, tuple[float, float, float]], SNES.ConvergedReason]
"""`SNES` convergence test callback."""

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
"""`TS` monitor callback."""

TSPreStepFunction = Callable[[TS], None]
"""`TS` pre-step callback."""

TSPostStepFunction = Callable[[TS], None]
"""`TS` post-step callback."""

TSEventHandlerFunction = Callable[[TS, float, Vec, NDArray[Scalar]], None]
"""`TS` event handler callback."""

TSPostEventFunction = Callable[[TS, NDArray[int], float, Vec, bool], None]
"""`TS` post-event handler callback."""

# --- TAO ---

TAOObjectiveFunction = Callable[[TAO, Vec], float]
"""`TAO` objective function callback."""

TAOGradientFunction = Callable[[TAO, Vec, Vec], None]
"""`TAO` objective gradient callback."""

TAOObjectiveGradientFunction =  Callable[[TAO, Vec, Vec], float]
"""`TAO` objective function and gradient callback."""

TAOHessianFunction = Callable[[TAO, Vec, Mat, Mat], None]
"""`TAO` objective Hessian callback."""

TAOUpdateFunction = Callable[[TAO, int], None]
"""`TAO` update callback."""

TAOMonitorFunction = Callable[[TAO], None]
"""`TAO` monitor callback."""

TAOConvergedFunction = Callable[[TAO], None]
"""`TAO` convergence test callback."""

TAOJacobianFunction = Callable[[TAO, Vec, Mat, Mat], None]
"""`TAO` Jacobian callback."""

TAOResidualFunction = Callable[[TAO, Vec, Vec], None]
"""`TAO` residual callback."""

TAOJacobianResidualFunction = Callable[[TAO, Vec, Mat, Mat], None]
"""`TAO` Jacobian residual callback."""

TAOVariableBoundsFunction = Callable[[TAO, Vec, Vec], None]
"""`TAO` variable bounds callback."""

TAOConstraintsFunction = Callable[[TAO, Vec, Vec], None]
"""`TAO` constraints callback."""
