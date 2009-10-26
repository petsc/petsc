
#if !defined(__DENSE_H)
#define __DENSE_H
#include "private/matimpl.h"


/*  
  MATSEQDENSE format - conventional dense Fortran storage (by columns)
*/

typedef struct {
  PetscScalar  *v;                /* matrix elements */
  PetscTruth   roworiented;       /* if true, row oriented input (default) */
  PetscInt     pad;               /* padding */        
  PetscBLASInt *pivots;           /* pivots in LU factorization */
  PetscBLASInt lda;               /* Lapack leading dimension of data */
  PetscTruth   changelda;         /* change lda on resize? Default unless user set lda */ 
  PetscBLASInt Mmax,Nmax;         /* indicates the largest dimensions of data possible */
  PetscTruth   user_alloc;        /* true if the user provided the dense data */
} Mat_SeqDense;

EXTERN PetscErrorCode MatMult_SeqDense(Mat A,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqDense(Mat A,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultTranspose_SeqDense(Mat A,Vec,Vec);
EXTERN PetscErrorCode MatMultTransposeAdd_SeqDense(Mat A,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMatMultSymbolic_SeqDense_SeqDense(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultNumeric_SeqDense_SeqDense(Mat,Mat,Mat);
EXTERN PetscErrorCode MatMatMultTranspose_SeqDense_SeqDense(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultTransposeSymbolic_SeqDense_SeqDense(Mat,Mat,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMultTransposeNumeric_SeqDense_SeqDense(Mat,Mat,Mat);
EXTERN PetscErrorCode MatMatMultSymbolic_SeqAIJ_SeqDense(Mat,Mat,PetscReal,Mat*); 
EXTERN PetscErrorCode MatMatMultNumeric_SeqAIJ_SeqDense(Mat,Mat,Mat);

EXTERN_C_BEGIN
EXTERN PetscErrorCode MatMatMult_SeqAIJ_SeqDense(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN PetscErrorCode MatMatMult_SeqDense_SeqDense(Mat,Mat,MatReuse,PetscReal,Mat*);
EXTERN_C_END

#endif
