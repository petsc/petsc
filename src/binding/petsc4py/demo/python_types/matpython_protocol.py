from petsc4py.typing import Scalar
from petsc4py.PETSc import Mat
from petsc4py.PETSc import Vec
from petsc4py.PETSc import IS
from petsc4py.PETSc import InsertMode
from petsc4py.PETSc import NormType
from petsc4py.PETSc import Viewer


# A template class with the Python methods supported by MATPYTHON

class MatPythonProtocol:

    def mult(self, A: Mat, x: Vec, y: Vec) -> None:
        """Matrix vector multiplication: y = A @ x."""
        ...

    def multAdd(self, A: Mat, x: Vec, y: Vec, z: Vec) -> None:
        """Matrix vector multiplication: z = A @ x + y."""
        ...

    def multTranspose(self, A: Mat, x: Vec, y: Vec) -> None:
        """Transposed matrix vector multiplication: y = A^T @ x."""
        ...

    def multTransposeAdd(self, A: Mat, x: Vec, y: Vec, z: Vec) -> None:
        """Transposed matrix vector multiplication: z = A^T @ x + y."""
        ...

    def multHermitian(self, A: Mat, x: Vec, y: Vec) -> None:
        """Hermitian matrix vector multiplication: y = A^H @ x."""
        ...

    def multHermitianAdd(self, A: Mat, x: Vec, y: Vec, z: Vec) -> None:
        """Hermitian matrix vector multiplication: z = A^H @ x + y."""
        ...

    def view(self, A: Mat, viewer: Viewer) -> None:
        """View the matrix."""
        ...

    def setFromOptions(self, A: Mat) -> None:
        """Process command line for customization."""
        ...

    def multDiagonalBlock(self, A: Mat, x: Vec, y: Vec) -> None:
        """Perform the on-process matrix vector multiplication."""
        ...

    def createVecs(self, A: Mat) -> tuple[Vec, Vec]:
        """Return tuple of vectors (x,y) suitable for A @ x = y."""
        ...

    def scale(self, A: Mat, s: Scalar) -> None:
        """Scale the matrix by a scalar."""
        ...

    def shift(self, A: Mat, s: Scalar) -> None:
        """Shift the matrix by a scalar."""
        ...

    def createSubMatrix(self, A: Mat, r: IS, c: IS, out: Mat) -> Mat:
        """Return the submatrix corresponding to r rows and c columns.

           Matrix out must be reused if not None.

        """
        ...

    def zeroRowsColumns(self, A: Mat, r: IS, diag: Scalar, x: Vec, b: Vec) -> None:
        """Zero rows and columns of the matrix corresponding to the index set r.

           Insert diag on the diagonal and modify vectors x and b accordingly if not None.

        """
        ...

    def getDiagonal(self, A: Mat, d: Vec) -> None:
        """Compute the diagonal of the matrix: d = diag(A)."""
        ...

    def setDiagonal(self, A: Mat, d: Vec, im: InsertMode) -> None:
        """Set the diagonal of the matrix."""
        ...

    def missingDiagonal(self, A: Mat, d: Vec, im: InsertMode) -> tuple[bool, int]:
        """Return a flag indicating if the matrix is missing a diagonal entry and the location."""
        ...

    def diagonalScale(self, A: Mat, L: Vec, R: Vec) -> None:
        """Perform left and right diagonal scaling if vectors are not None.

        A = diag(L)@A@diag(R).

        """
        ...

    def getDiagonalBlock(self, A: Mat) -> Mat:
        """Return the on-process matrix."""
        ...

    def setUp(self, A: Mat) -> None:
        """Perform the required setup."""
        ...

    def duplicate(self, A: Mat, op: Mat.DuplicateOption) -> Mat:
        """Duplicate the matrix."""
        ...

    def copy(self, A: Mat, B: Mat, op: Mat.Structure) -> None:
        """Copy the matrix: B = A."""
        ...

    def productSetFromOptions(self, A: Mat, prodtype: str, X: Mat, Y: Mat, Z: Mat) -> bool:
        """The boolean flag indicating if the matrix supports prodtype."""
        ...

    def productSymbolic(self, A: Mat, product: Mat, producttype: str, X: Mat, Y: Mat, Z: Mat) -> None:
        """Perform the symbolic stage of the requested matrix product."""
        ...

    def productNumeric(self, A: Mat, product: Mat, producttype: str, X: Mat, Y: Mat, Z: Mat) -> None:
        """Perform the numeric stage of the requested matrix product."""
        ...

    def zeroEntries(self, A: Mat) -> None:
        """Set the matrix to zero."""
        ...

    def norm(self, A: Mat, normtype: NormType) -> float:
        """Compute the norm of the matrix."""
        ...

    def solve(self, A: Mat, y: Vec, x: Vec) -> None:
        """Solve the equation: x = inv(A) y."""
        ...

    def solveAdd(self, A: Mat, y: Vec, z: Vec, x: Vec) -> None:
        """Solve the equation: x = inv(A) y + z."""
        ...

    def solveTranspose(self, A: Mat, y: Vec, x: Vec) -> None:
        """Solve the equation: x = inv(A)^T y."""
        ...

    def solveTransposeAdd(self, A: Mat, y: Vec, z: Vec, x: Vec) -> None:
        """Solve the equation: x = inv(A)^T y + z."""
        ...

    def SOR(self, A: Mat, b: Vec, omega: float, sortype: Mat.SORType,
            shift: float, its: int, lits: int, x: Vec) -> None:
        """Perform SOR iterations."""
        ...

    def conjugate(self, A: Mat) -> None:
        """Perform the conjugation of the matrix: A = conj(A)."""
        ...

    def imagPart(self, A: Mat) -> None:
        """Set real part to zero. A = imag(A)."""
        ...

    def realPart(self, A: sMat) -> None:
        """Set imaginary part to zero. A = real(A)."""
        ...
