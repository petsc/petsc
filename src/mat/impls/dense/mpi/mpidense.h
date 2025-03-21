#pragma once

#include <../src/mat/impls/dense/seq/dense.h>
#include <petscsf.h>

/*  Data structures for basic parallel dense matrix  */

typedef struct {  /* used by MatMatMultxxx_MPIDense_MPIDense() */
  Mat Ae, Be, Ce; /* matrix in Elemental format */
} Mat_MatMultDense;

typedef struct { /* used by MatTransposeMatMultXXX_MPIDense_MPIDense() */
  PetscScalar *sendbuf;
  Mat          atb;
  PetscMPIInt *recvcounts;
  PetscMPIInt  tag;
} Mat_TransMatMultDense;

typedef struct { /* used by MatMatTransposeMultxxx_MPIDense_MPIDense() */
  PetscScalar *buf[2];
  PetscMPIInt  tag;
  PetscMPIInt *recvcounts;
  PetscMPIInt *recvdispls;
  PetscInt     alg; /* algorithm used */
} Mat_MatTransMultDense;

PETSC_INTERN PetscErrorCode MatSetUpMultiply_MPIDense(Mat);
PETSC_INTERN PetscErrorCode MatCreateSubMatrices_MPIDense(Mat, PetscInt, const IS[], const IS[], MatReuse, Mat *[]);
PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIAIJ_MPIDense(Mat);

PETSC_INTERN PetscErrorCode MatProductSetFromOptions_MPIDense_MPIAIJ(Mat);

#if defined(PETSC_HAVE_ELEMENTAL)
PETSC_INTERN PetscErrorCode MatMatMultSymbolic_Elemental(Mat, Mat, PetscReal, Mat);
PETSC_INTERN PetscErrorCode MatMatMultNumeric_Elemental(Mat, Mat, Mat);
#endif
PETSC_INTERN PetscErrorCode MatConvert_SeqDense_SeqAIJ(Mat A, MatType newtype, MatReuse reuse, Mat *newmat);

PETSC_INTERN PetscErrorCode MatShift_MPIDense(Mat, PetscScalar);
PETSC_INTERN PetscErrorCode MatDenseGetColumnVecWrite_MPIDense(Mat, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode MatDenseGetColumnVecRead_MPIDense(Mat, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode MatDenseGetColumnVec_MPIDense(Mat, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode MatDenseRestoreColumnVecWrite_MPIDense(Mat, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode MatDenseRestoreColumnVecRead_MPIDense(Mat, PetscInt, Vec *);
PETSC_INTERN PetscErrorCode MatDenseRestoreColumnVec_MPIDense(Mat, PetscInt, Vec *);

PETSC_INTERN PetscErrorCode MatCreate_MPIDense(Mat);
PETSC_INTERN PetscErrorCode MatGetDiagonal_MPIDense(Mat, Vec);

#if PetscDefined(HAVE_CUDA)
PETSC_INTERN PetscErrorCode MatConvert_MPIDense_MPIDenseCUDA(Mat, MatType, MatReuse, Mat *);
#endif

#if PetscDefined(HAVE_HIP)
PETSC_INTERN PetscErrorCode MatConvert_MPIDense_MPIDenseHIP(Mat, MatType, MatReuse, Mat *);
#endif
