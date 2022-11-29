#include <petscmat.h>
#include <petsc/private/matorderimpl.h>

#if defined(PETSC_HAVE_SUPERLU_DIST)

  /* SuperLU_DIST bundles f2ced mc64ad_() from HSL */

  /*
   SuperLU_dist uses a common flag for both Fortran mangling and BLAS/LAPACK mangling which
   corresponds to the PETSc BLAS/LAPACK mangling flag (we pass this flag to configure SuperLU_dist)
*/

  /* Why not include superlu_dist inludes? */
  #if defined(PETSC_BLASLAPACK_CAPS)
    #define mc64id_dist MC64ID_DIST
    #define mc64ad_dist MC64AD_DIST

  #elif !defined(PETSC_BLASLAPACK_UNDERSCORE)
    #define mc64id_dist mc64id_dist
    #define mc64ad_dist mc64ad_dist

  #endif

PETSC_EXTERN PetscInt mc64id_dist(PetscInt *);
PETSC_EXTERN PetscInt mc64ad_dist(const PetscInt *, PetscInt *, PetscInt *, const PetscInt *, const PetscInt *n, PetscScalar *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscInt *, PetscScalar *, PetscInt *, PetscInt *);
#endif

/*
  MatGetOrdering_WBM - Find the nonsymmetric reordering of the graph which maximizes the product of diagonal entries,
    using weighted bipartite graph matching. This is MC64 in the Harwell-Boeing library.
*/
PETSC_INTERN PetscErrorCode MatGetOrdering_WBM(Mat mat, MatOrderingType type, IS *row, IS *col)
{
  PetscScalar    *a, *dw;
  const PetscInt *ia, *ja;
  PetscInt        job = 5;
  PetscInt       *perm, nrow, ncol, nnz, liw, *iw, ldw;
  PetscBool       done;
#if defined(PETSC_HAVE_SUPERLU_DIST)
  PetscInt num, info[10], icntl[10], i;
#endif

  PetscFunctionBegin;
  PetscCall(MatGetRowIJ(mat, 1, PETSC_TRUE, PETSC_TRUE, &nrow, &ia, &ja, &done));
  ncol = nrow;
  nnz  = ia[nrow];
  PetscCheck(done, PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "Cannot get rows for matrix");
  PetscCall(MatSeqAIJGetArray(mat, &a));
  switch (job) {
  case 1:
    liw = 4 * nrow + ncol;
    ldw = 0;
    break;
  case 2:
    liw = 2 * nrow + 2 * ncol;
    ldw = ncol;
    break;
  case 3:
    liw = 8 * nrow + 2 * ncol + nnz;
    ldw = nnz;
    break;
  case 4:
    liw = 3 * nrow + 2 * ncol;
    ldw = 2 * ncol + nnz;
    break;
  case 5:
    liw = 3 * nrow + 2 * ncol;
    ldw = nrow + 2 * ncol + nnz;
    break;
  }

  PetscCall(PetscMalloc3(liw, &iw, ldw, &dw, nrow, &perm));
#if defined(PETSC_HAVE_SUPERLU_DIST)
  PetscCallExternal(mc64id_dist, icntl);
  icntl[0] = 0;  /* allow printing error messages (f2c'd code uses if non-negative, ignores value otherwise) */
  icntl[1] = -1; /* suppress warnings */
  icntl[2] = -1; /* ignore diagnostic output [default] */
  icntl[3] = 0;  /* perform consistency checks [default] */
  PetscCallExternal(mc64ad_dist, &job, &nrow, &nnz, ia, ja, a, &num, perm, &liw, iw, &ldw, dw, icntl, info);
  PetscCall(MatRestoreRowIJ(mat, 1, PETSC_TRUE, PETSC_TRUE, NULL, &ia, &ja, &done));
  for (i = 0; i < nrow; ++i) perm[i]--;
  /* If job == 5, dw[0..ncols] contains the column scaling and dw[ncols..ncols+nrows] contains the row scaling */
  PetscCall(ISCreateStride(PETSC_COMM_SELF, nrow, 0, 1, row));
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, nrow, perm, PETSC_COPY_VALUES, col));
  PetscCall(PetscFree3(iw, dw, perm));
  PetscFunctionReturn(0);
#else
  SETERRQ(PetscObjectComm((PetscObject)mat), PETSC_ERR_SUP, "WBM using MC64 does not support complex numbers");
#endif
}
