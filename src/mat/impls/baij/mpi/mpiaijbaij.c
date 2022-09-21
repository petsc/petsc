
#include <../src/mat/impls/baij/mpi/mpibaij.h> /*I "petscmat.h" I*/
#include <../src/mat/impls/aij/mpi/mpiaij.h>
#include <petsc/private/matimpl.h>
#include <petscmat.h>

PETSC_INTERN PetscErrorCode MatConvert_MPIAIJ_MPIBAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat)
{
  Mat         M;
  Mat_MPIAIJ *mpimat = (Mat_MPIAIJ *)A->data;
  Mat_SeqAIJ *Aa = (Mat_SeqAIJ *)mpimat->A->data, *Ba = (Mat_SeqAIJ *)mpimat->B->data;
  PetscInt   *d_nnz, *o_nnz;
  PetscInt    i, m, n, lm, ln, bs = PetscAbs(A->rmap->bs);

  PetscFunctionBegin;
  if (reuse != MAT_REUSE_MATRIX) {
    PetscCall(MatGetSize(A, &m, &n));
    PetscCall(MatGetLocalSize(A, &lm, &ln));
    PetscCall(PetscMalloc2(lm / bs, &d_nnz, lm / bs, &o_nnz));

    for (i = 0; i < lm / bs; i++) {
      d_nnz[i] = (Aa->i[i * bs + 1] - Aa->i[i * bs]) / bs;
      o_nnz[i] = (Ba->i[i * bs + 1] - Ba->i[i * bs]) / bs;
    }

    PetscCall(MatCreate(PetscObjectComm((PetscObject)A), &M));
    PetscCall(MatSetSizes(M, lm, ln, m, n));
    PetscCall(MatSetType(M, MATMPIBAIJ));
    PetscCall(MatSeqBAIJSetPreallocation(M, bs, 0, d_nnz));
    PetscCall(MatMPIBAIJSetPreallocation(M, bs, 0, d_nnz, 0, o_nnz));
    PetscCall(PetscFree2(d_nnz, o_nnz));
  } else M = *newmat;

  /* reuse may not be equal to MAT_REUSE_MATRIX, but the basic converter will reallocate or replace newmat if this value is not used */
  /* if reuse is equal to MAT_INITIAL_MATRIX, it has been appropriately preallocated before                                          */
  /*                      MAT_INPLACE_MATRIX, it will be replaced with MatHeaderReplace below                                        */
  PetscCall(MatConvert_Basic(A, newtype, MAT_REUSE_MATRIX, &M));

  if (reuse == MAT_INPLACE_MATRIX) {
    PetscCall(MatHeaderReplace(A, &M));
  } else *newmat = M;
  PetscFunctionReturn(0);
}
