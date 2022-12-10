
#include <../src/mat/impls/sbaij/mpi/mpisbaij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/matimpl.h>
#include <petscmat.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqBAIJ_Preallocate(Mat, PetscInt **);
PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqSBAIJ_Preallocate(Mat, PetscInt **);

/* The code is virtually identical to MatConvert_MPIAIJ_MPIBAIJ() */
PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPISBAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat         M;
  Mat_MPIAIJ *mpimat = (Mat_MPIAIJ *)A->data;
  PetscInt   *d_nnz, *o_nnz;
  PetscInt    m, n, lm, ln, bs = PetscAbs(A->rmap->bs);

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatDisAssemble_MPIAIJ(A));
    PetscCall(MatGetSize(A, &m, &n));
    PetscCall(MatGetLocalSize(A, &lm, &ln));
    PetscCall(MatConvert_SeqAIJ_SeqSBAIJ_Preallocate(mpimat->A, &d_nnz));
    PetscCall(MatConvert_SeqAIJ_SeqBAIJ_Preallocate(mpimat->B, &o_nnz));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &M));
    PetscCall(MatSetSizes(M, lm, ln, m, n));
    PetscCall(MatSetType(M, MATMPISBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(M, bs, 0, d_nnz));
    PetscCall(MatMPISBAIJSetPreallocation(M, bs, 0, d_nnz, 0, o_nnz));
    PetscCall(PetscFree(d_nnz));
    PetscCall(PetscFree(o_nnz));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  } else M = *newmat;

  /* reuse may not be equal to MAT_REUSE_MATRIX, but the basic converter will reallocate or replace newmat if this value is not used */
  /* if reuse is equal to MAT_INITIAL_MATRIX, it has been appropriately preallocated before                                          */
  /*                      MAT_INPLACE_MATRIX, it will be replaced with MatHeaderReplace below                                        */
  PetscCall(MatConvert_Basic(A, newtype, MAT_REUSE_MATRIX, &M));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &M));
  } else *newmat = M;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* contributed by Dahai Guo <dhguo@ncsa.uiuc.edu> April 2011 */
PETSC_INTERN PetscErrorCode MatConvert_MPIBAIJ_MPISBAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat                M;
  Mat_MPIBAIJ       *mpimat = (Mat_MPIBAIJ *)A->data;
  Mat_SeqBAIJ       *Aa = (Mat_SeqBAIJ *)mpimat->A->data, *Ba = (Mat_SeqBAIJ *)mpimat->B->data;
  PetscInt          *d_nnz, *o_nnz;
  PetscInt           i, nz;
  PetscInt           m, n, lm, ln;
  PetscInt           rstart, rend;
  const PetscScalar *vwork;
  const PetscInt    *cwork;
  PetscInt           bs = A->rmap->bs;

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatGetSize(A, &m, &n));
    PetscCall(MatGetLocalSize(A, &lm, &ln));
    PetscCall(PetscMalloc2(lm / bs, &d_nnz, lm / bs, &o_nnz));

    PetscCall(MatMarkDiagonal_SeqBAIJ(mpimat->A));
    for (i = 0; i < lm / bs; i++) {
      d_nnz[i] = Aa->i[i + 1] - Aa->diag[i];
      o_nnz[i] = Ba->i[i + 1] - Ba->i[i];
    }

    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &M));
    PetscCall(MatSetSizes(M, lm, ln, m, n));
    PetscCall(MatSetType(M, MATMPISBAIJ));
    PetscCall(MatSeqSBAIJSetPreallocation(M, bs, 0, d_nnz));
    PetscCall(MatMPISBAIJSetPreallocation(M, bs, 0, d_nnz, 0, o_nnz));

    PetscCall(PetscFree2(d_nnz, o_nnz));
  } else M = *newmat;

  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  PetscCall(MatSetOption(M, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));
  for (i = rstart; i < rend; i++) {
    PetscCall(MatGetRow(A, i, &nz, &cwork, &vwork));
    PetscCall(MatSetValues(M, 1, &i, nz, cwork, vwork, INSERT_VALUES));
    PetscCall(MatRestoreRow(A, i, &nz, &cwork, &vwork));
  }
  PetscCall(MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &M));
  } else *newmat = M;
  PetscFunctionReturn(PETSC_SUCCESS);
}
