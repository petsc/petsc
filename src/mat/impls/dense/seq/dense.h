
#if !defined(__DENSE_H)
#define __DENSE_H
#include "src/mat/matimpl.h"


/*  
  MATSEQDENSE format - conventional dense Fortran storage (by columns)
*/

typedef struct {
  PetscScalar  *v;                /* matrix elements */
  PetscTruth   roworiented;       /* if true, row oriented input (default) */
  PetscInt     pad;               /* padding */        
  PetscBLASInt *pivots;           /* pivots in LU factorization */
  PetscBLASInt lda;               /* Lapack leading dimension of data */
  PetscBLASInt Nmax;              /* indicates the second dimension of data */
  PetscTruth   user_alloc;        /* true if the user provided the dense data */
} Mat_SeqDense;

EXTERN PetscErrorCode MatMult_SeqDense(Mat A,Vec,Vec);
EXTERN PetscErrorCode MatMultAdd_SeqDense(Mat A,Vec,Vec,Vec);
EXTERN PetscErrorCode MatMultTranspose_SeqDense(Mat A,Vec,Vec);
EXTERN PetscErrorCode MatMultTransposeAdd_SeqDense(Mat A,Vec,Vec,Vec);

#endif
