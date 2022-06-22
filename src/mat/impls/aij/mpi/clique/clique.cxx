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
PetscErrorCode MatConvertToSparseElemental(Mat A,MatReuse reuse,Mat_SparseElemental *cliq)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatView_SparseElemental(Mat A,PetscViewer viewer)
{
  PetscBool iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscViewerFormat format;
    PetscCall(PetscViewerGetFormat(viewer,&format));
    if (format == PETSC_VIEWER_ASCII_INFO) {
      PetscCall(PetscViewerASCIIPrintf(viewer,"SparseElemental run parameters:\n"));
    } else if (format == PETSC_VIEWER_DEFAULT) { /* matrix A is factored matrix, remove this block */
      Mat Aaij;
      PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_FALSE));
      PetscCall(PetscViewerASCIIUseTabs(viewer,PETSC_TRUE));
      PetscCall(PetscPrintf(PetscObjectComm((PetscObject)viewer),"SparseElemental matrix\n"));
      PetscCall(MatComputeOperator(A,MATAIJ,&Aaij));
      PetscCall(MatView(Aaij,viewer));
      PetscCall(MatDestroy(&Aaij));
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode MatDestroy_SparseElemental(Mat A)
{
  PetscFunctionBegin;
  PetscCall(PetscObjectComposeFunction((PetscObject)A,"MatFactorGetSolverType_C",NULL));
  PetscFunctionReturn(0);
}

PetscErrorCode MatSolve_SparseElemental(Mat A,Vec B,Vec X)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorNumeric_SparseElemental(Mat F,Mat A,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PetscErrorCode MatCholeskyFactorSymbolic_SparseElemental(Mat F,Mat A,IS r,const MatFactorInfo *info)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

/*MC
     MATSOLVERSPARSEELEMENTAL  - A solver package providing direct solvers for sparse distributed
  and sequential matrices via the external package Elemental

  Use ./configure --download-elemental to have PETSc installed with Elemental

  Use -pc_type lu -pc_factor_mat_solver_type sparseelemental to use this direct solver

  This is currently not supported.

  Developer Note: Jed Brown made the interface for Clique when it was a standalone package. Later Jack Poulson merged and refactored Clique into
  Elemental but since the Clique interface was not tested in PETSc the interface was not updated for the new Elemental interface. Later Barry Smith updated
  all the boilerplate for the Clique interface to SparseElemental but since the solver interface changed dramatically he did not update the code
  that actually calls the SparseElemental solvers. We are waiting on someone who has a need to complete the SparseElemental interface from PETSc.

  Level: beginner

.seealso: `PCFactorSetMatSolverType()`, `MatSolverType`

M*/

PetscErrorCode MatFactorGetSolverType_SparseElemental(Mat A,MatSolverType *type)
{
  PetscFunctionBegin;
  *type = MATSOLVERSPARSEELEMENTAL;
  PetscFunctionReturn(0);
}

static PetscErrorCode MatGetFactor_aij_sparseelemental(Mat A,MatFactorType ftype,Mat *F)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode MatSolverTypeRegister_SparseElemental(void)
{
  PetscFunctionBegin;
  PetscCall(MatSolverTypeRegister(MATSOLVERSPARSEELEMENTAL,MATMPIAIJ,MAT_FACTOR_LU,MatGetFactor_aij_sparseelemental));
  PetscFunctionReturn(0);
}
