
#include <../src/mat/impls/baij/mpi/mpibaij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/matimpl.h>
#include <petscmat.h>

PETSC_INTERN PetscErrorCode MatConvert_SeqAIJ_SeqBAIJ_Preallocate(Mat, PetscInt **);

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIBAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat         M;
  Mat_MPIAIJ *mpimat = (Mat_MPIAIJ *)A->data;
  PetscInt   *d_nnz, *o_nnz;
  PetscInt    m, n, lm, ln, bs = PetscAbs(A->rmap->bs);

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    PetscBool3 sym = A->symmetric, hermitian = A->hermitian, structurally_symmetric = A->structurally_symmetric, spd = A->spd;
    PetscCall(MatDisAssemble_MPIAIJ(A));
    PetscCall(MatGetSize(A, &m, &n));
    PetscCall(MatGetLocalSize(A, &lm, &ln));
    PetscCall(MatConvert_SeqAIJ_SeqBAIJ_Preallocate(mpimat->A, &d_nnz));
    PetscCall(MatConvert_SeqAIJ_SeqBAIJ_Preallocate(mpimat->B, &o_nnz));
    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &M));
    PetscCall(MatSetSizes(M, lm, ln, m, n));
    PetscCall(MatSetType(M, MATMPIBAIJ));
    PetscCall(MatSeqBAIJSetPreallocation(M, bs, 0, d_nnz));
    PetscCall(MatMPIBAIJSetPreallocation(M, bs, 0, d_nnz, 0, o_nnz));
    PetscCall(PetscFree(d_nnz));
    PetscCall(PetscFree(o_nnz));
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    A->symmetric              = sym;
    A->hermitian              = hermitian;
    A->structurally_symmetric = structurally_symmetric;
    A->spd                    = spd;
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
