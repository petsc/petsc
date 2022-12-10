#include <petscmat.h>
#include <adolc/adolc_sparse.h>

/*
   REQUIRES configuration of PETSc with option --download-adolc.

   For documentation on ADOL-C, see
     $PETSC_ARCH/externalpackages/ADOL-C-2.6.0/ADOL-C/doc/adolc-manual.pdf
*/

/*
  Basic printing for sparsity pattern

  Input parameters:
  comm     - MPI communicator
  m        - number of rows

  Output parameter:
  sparsity - matrix sparsity pattern, typically computed using an ADOL-C function such as jac_pat
*/
PetscErrorCode PrintSparsity(MPI_Comm comm, PetscInt m, unsigned int **sparsity)
{
  PetscFunctionBegin;
  PetscCall(PetscPrintf(comm, "Sparsity pattern:\n"));
  for (PetscInt i = 0; i < m; i++) {
    PetscCall(PetscPrintf(comm, "\n %2d: ", i));
    for (PetscInt j = 1; j <= (PetscInt)sparsity[i][0]; j++) PetscCall(PetscPrintf(comm, " %2d ", sparsity[i][j]));
  }
  PetscCall(PetscPrintf(comm, "\n\n"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Generate a seed matrix defining the partition of columns of a matrix by a particular coloring,
  used for matrix compression

  Input parameter:
  iscoloring - the index set coloring to be used

  Output parameter:
  S          - the resulting seed matrix

  Notes:
  Before calling GenerateSeedMatrix, Seed should be allocated as a logically 2d array with number of
  rows equal to the matrix to be compressed and number of columns equal to the number of colors used
  in iscoloring.
*/
PetscErrorCode GenerateSeedMatrix(ISColoring iscoloring, PetscScalar **S)
{
  IS             *is;
  PetscInt        p, size;
  const PetscInt *indices;

  PetscFunctionBegin;
  PetscCall(ISColoringGetIS(iscoloring, PETSC_USE_POINTER, &p, &is));
  for (PetscInt colour = 0; colour < p; colour++) {
    PetscCall(ISGetLocalSize(is[colour], &size));
    PetscCall(ISGetIndices(is[colour], &indices));
    for (PetscInt j = 0; j < size; j++) S[indices[j]][colour] = 1.;
    PetscCall(ISRestoreIndices(is[colour], &indices));
  }
  PetscCall(ISColoringRestoreIS(iscoloring, PETSC_USE_POINTER, &is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Establish a look-up vector whose entries contain the colour used for that diagonal entry. Clearly
  we require the number of dependent and independent variables to be equal in this case.
  Input parameters:
  S        - the seed matrix defining the coloring
  sparsity - the sparsity pattern of the matrix to be recovered, typically computed using an ADOL-C
             function, such as jac_pat or hess_pat
  m        - the number of rows of Seed (and the matrix to be recovered)
  p        - the number of colors used (also the number of columns in Seed)
  Output parameter:
  R        - the recovery vector to be used for de-compression
*/
PetscErrorCode GenerateSeedMatrixPlusRecovery(ISColoring iscoloring, PetscScalar **S, PetscScalar *R)
{
  IS             *is;
  PetscInt        p, size, colour, j;
  const PetscInt *indices;

  PetscFunctionBegin;
  PetscCall(ISColoringGetIS(iscoloring, PETSC_USE_POINTER, &p, &is));
  for (colour = 0; colour < p; colour++) {
    PetscCall(ISGetLocalSize(is[colour], &size));
    PetscCall(ISGetIndices(is[colour], &indices));
    for (j = 0; j < size; j++) {
      S[indices[j]][colour] = 1.;
      R[indices[j]]         = colour;
    }
    PetscCall(ISRestoreIndices(is[colour], &indices));
  }
  PetscCall(ISColoringRestoreIS(iscoloring, PETSC_USE_POINTER, &is));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Establish a look-up matrix whose entries contain the column coordinates of the corresponding entry
  in a matrix which has been compressed using the coloring defined by some seed matrix

  Input parameters:
  S        - the seed matrix defining the coloring
  sparsity - the sparsity pattern of the matrix to be recovered, typically computed using an ADOL-C
             function, such as jac_pat or hess_pat
  m        - the number of rows of Seed (and the matrix to be recovered)
  p        - the number of colors used (also the number of columns in Seed)

  Output parameter:
  R        - the recovery matrix to be used for de-compression
*/
PetscErrorCode GetRecoveryMatrix(PetscScalar **S, unsigned int **sparsity, PetscInt m, PetscInt p, PetscScalar **R)
{
  PetscInt i, j, k, colour;

  PetscFunctionBegin;
  for (i = 0; i < m; i++) {
    for (colour = 0; colour < p; colour++) {
      R[i][colour] = -1.;
      for (k = 1; k <= (PetscInt)sparsity[i][0]; k++) {
        j = (PetscInt)sparsity[i][k];
        if (S[j][colour] == 1.) {
          R[i][colour] = j;
          break;
        }
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Recover the values of a sparse matrix from a compressed format and insert these into a matrix

  Input parameters:
  mode - use INSERT_VALUES or ADD_VALUES, as required
  m    - number of rows of matrix.
  p    - number of colors used in the compression of J (also the number of columns of R)
  R    - recovery matrix to use in the decompression procedure
  C    - compressed matrix to recover values from
  a    - shift value for implicit problems (select NULL or unity for explicit problems)

  Output parameter:
  A    - Mat to be populated with values from compressed matrix
*/
PetscErrorCode RecoverJacobian(Mat A, InsertMode mode, PetscInt m, PetscInt p, PetscScalar **R, PetscScalar **C, PetscReal *a)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < m; i++) {
    for (PetscInt colour = 0; colour < p; colour++) {
      PetscInt j = (PetscInt)R[i][colour];
      if (j != -1) {
        if (a) C[i][colour] *= *a;
        PetscCall(MatSetValues(A, 1, &i, 1, &j, &C[i][colour], mode));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Recover the values of the local portion of a sparse matrix from a compressed format and insert
  these into the local portion of a matrix

  Input parameters:
  mode - use INSERT_VALUES or ADD_VALUES, as required
  m    - number of rows of matrix.
  p    - number of colors used in the compression of J (also the number of columns of R)
  R    - recovery matrix to use in the decompression procedure
  C    - compressed matrix to recover values from
  a    - shift value for implicit problems (select NULL or unity for explicit problems)

  Output parameter:
  A    - Mat to be populated with values from compressed matrix
*/
PetscErrorCode RecoverJacobianLocal(Mat A, InsertMode mode, PetscInt m, PetscInt p, PetscScalar **R, PetscScalar **C, PetscReal *a)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < m; i++) {
    for (PetscInt colour = 0; colour < p; colour++) {
      PetscInt j = (PetscInt)R[i][colour];
      if (j != -1) {
        if (a) C[i][colour] *= *a;
        PetscCall(MatSetValuesLocal(A, 1, &i, 1, &j, &C[i][colour], mode));
      }
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Recover the diagonal of the Jacobian from its compressed matrix format

  Input parameters:
  mode - use INSERT_VALUES or ADD_VALUES, as required
  m    - number of rows of matrix.
  R    - recovery vector to use in the decompression procedure
  C    - compressed matrix to recover values from
  a    - shift value for implicit problems (select NULL or unity for explicit problems)

  Output parameter:
  diag - Vec to be populated with values from compressed matrix
*/
PetscErrorCode RecoverDiagonal(Vec diag, InsertMode mode, PetscInt m, PetscScalar *R, PetscScalar **C, PetscReal *a)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < m; i++) {
    PetscInt colour = (PetscInt)R[i];
    if (a) C[i][colour] *= *a;
    PetscCall(VecSetValues(diag, 1, &i, &C[i][colour], mode));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Recover the local portion of the diagonal of the Jacobian from its compressed matrix format

  Input parameters:
  mode - use INSERT_VALUES or ADD_VALUES, as required
  m    - number of rows of matrix.
  R    - recovery vector to use in the decompression procedure
  C    - compressed matrix to recover values from
  a    - shift value for implicit problems (select NULL or unity for explicit problems)

  Output parameter:
  diag - Vec to be populated with values from compressed matrix
*/
PetscErrorCode RecoverDiagonalLocal(Vec diag, InsertMode mode, PetscInt m, PetscScalar *R, PetscScalar **C, PetscReal *a)
{
  PetscFunctionBegin;
  for (PetscInt i = 0; i < m; i++) {
    PetscInt colour = (PetscInt)R[i];
    if (a) C[i][colour] *= *a;
    PetscCall(VecSetValuesLocal(diag, 1, &i, &C[i][colour], mode));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
