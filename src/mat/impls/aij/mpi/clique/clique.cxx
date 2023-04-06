#include <../src/mat/impls/aij/mpi/clique/matcliqueimpl.h> /*I "petscmat.h" I*/

/*
  MatConvertToSparseElemental: Convert Petsc aij matrix to sparse elemental matrix

  input:
+   A     - matrix in seqaij or mpiaij format
-   reuse - denotes if the destination matrix is to be created or reused.
            Use MAT_INPLACE_MATRIX for inplace conversion, otherwise use MAT_INITIAL_MATRIX or MAT_REUSE_MATRIX.

  output:
.   cliq - Clique context
*/
PetscErrorCode MatConvertToSparseElemental(Mat, MatReuse, Mat_SparseElemental *)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatView_SparseElemental(Mat A, PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) {
    PetscViewerFormat format;
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscCall(PetscViewerASCIIPrintf(viewer, "SparseElemental run parameters:\n"));
    } else if (format == PETSC_VIEWER_DEFAULT) { /* matrix A is factored matrix, remove this block */
      Mat Aaij;
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_FALSE));
      PetscCall(PetscViewerASCIIUseTabs(viewer, PETSC_TRUE));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)viewer), "SparseElemental matrix\n"));
      PetscCall(MatComputeOperator(A, MATAIJ, &Aaij));
      PetscCall(MatView(Aaij, viewer));
      PetscCall(MatDestroy(&Aaij));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatDestroy_SparseElemental(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)A, "MatFactorGetSolverType_C", nullptr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatSolve_SparseElemental(Mat, Vec, Vec)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCholeskyFactorNumeric_SparseElemental(Mat, Mat, const MatFactorInfo *)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode MatCholeskyFactorSymbolic_SparseElemental(Mat, Mat, IS, const MatFactorInfo *)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*MC
     MATSOLVERSPARSEELEMENTAL  - A solver package providing direct solvers for sparse distributed
  and sequential matrices via the external package Elemental

  Use ./configure --download-elemental to have PETSc installed with Elemental

  Use -pc_type lu -pc_factor_mat_solver_type sparseelemental to use this direct solver

  This is currently not supported.

  Developer Note:
  Jed Brown made the interface for Clique when it was a standalone package. Later Jack Poulson merged and refactored Clique into
  Elemental but since the Clique interface was not tested in PETSc the interface was not updated for the new Elemental interface. Later Barry Smith updated
  all the boilerplate for the Clique interface to SparseElemental but since the solver interface changed dramatically he did not update the code
  that actually calls the SparseElemental solvers. We are waiting on someone who has a need to complete the SparseElemental interface from PETSc.

  Level: beginner

.seealso: [](chapter_matrices), `Mat`, `PCFactorSetMatSolverType()`, `MatSolverType`
M*/

PetscErrorCode MatFactorGetSolverType_SparseElemental(Mat, MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSPARSEELEMENTAL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode MatGetFactor_aij_sparseelemental(Mat, MatFactorType, Mat *)
{
  PetscFunctionBegin;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_SparseElemental(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERSPARSEELEMENTAL, MATMPIAIJ, MAT_FACTOR_LU, MatGetFactor_aij_sparseelemental));
  PetscFunctionReturn(PETSC_SUCCESS);
}
