
#if !defined(__DENSE_H)
#define __DENSE_H
#include <petsc/private/matimpl.h>
#include <../src/mat/impls/aij/seq/aij.h> /* Mat_MatTransMatMult is defined here */

/*
  MATSEQDENSE format - conventional dense Fortran storage (by columns)
*/

typedef struct {
  PetscScalar  *v;                /* matrix elements */
  PetscScalar  *unplacedarray;    /* if one called MatDensePlaceArray(), this is where it stashed the original */
  PetscBool    roworiented;       /* if true, row oriented input (default) */
  PetscInt     pad;               /* padding */
  PetscBLASInt *pivots;           /* pivots in LU factorization */
  PetscBLASInt lfwork;            /* length of work array in factorization */
  PetscScalar  *fwork;            /* work array in factorization */
  PetscBLASInt lda;               /* Lapack leading dimension of data */
  PetscBool    changelda;         /* change lda on resize? Default unless user set lda */
  PetscBLASInt Mmax,Nmax;         /* indicates the largest dimensions of data possible */
  PetscBool    user_alloc;        /* true if the user provided the dense data */
  PetscBool    unplaced_user_alloc; 
  Mat          ptapwork;          /* workspace (SeqDense matrix) for PtAP */

  Mat_MatTransMatMult *atb;       /* used by MatTransposeMatMult_SeqAIJ_SeqDense */
} Mat_SeqDense;

extern PetscErrorCode MatMatMultSymbolic_SeqDense_SeqDense(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatMatMultNumeric_SeqDense_SeqDense(Mat,Mat,Mat);
extern PetscErrorCode MatTransposeMatMult_SeqDense_SeqDense(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode MatTransposeMatMultSymbolic_SeqDense_SeqDense(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatTransposeMatMultNumeric_SeqDense_SeqDense(Mat,Mat,Mat);
extern PetscErrorCode MatMatTransposeMult_SeqDense_SeqDense(Mat,Mat,MatReuse,PetscReal,Mat*);
extern PetscErrorCode MatMatTransposeMultSymbolic_SeqDense_SeqDense(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatMatTransposeMultNumeric_SeqDense_SeqDense(Mat,Mat,Mat);
extern PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqDense(Mat,Mat,PetscReal,Mat*);
extern PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqDense(Mat,Mat,Mat);

PETSC_INTERN PetscErrorCode MatDestroy_SeqDense_MatTransMatMult(Mat);

PETSC_INTERN PetscErrorCode MatMatMult_SeqAIJ_SeqDense(Mat,Mat,MatReuse,PetscReal,Mat*);
PETSC_INTERN PetscErrorCode MatMatMult_SeqDense_SeqDense(Mat,Mat,MatReuse,PetscReal,Mat*);
PETSC_EXTERN PetscErrorCode MatSeqDenseInvertFactors_Private(Mat);

#endif
