/* $Id: dense.h,v 1.7 2000/01/11 21:00:34 bsmith Exp bsmith $ */

#include "src/mat/matimpl.h"

#if !defined(__DENSE_H)
#define __DENSE_H

/*  
  MATSEQDENSE format - conventional dense Fortran storage (by columns)
*/

typedef struct {
  Scalar *v;                /* matrix elements */
  int    roworiented;       /* if true, row oriented input (default) */
  int    m,n;              /* rows, columns */
  int    pad;               /* padding */        
  int    *pivots;           /* pivots in LU factorization */
  int    user_alloc;        /* true if the user provided the dense data */
} Mat_SeqDense;

EXTERN int MatMult_SeqDense(Mat A,Vec,Vec);
EXTERN int MatMultAdd_SeqDense(Mat A,Vec,Vec,Vec);
EXTERN int MatMultTranspose_SeqDense(Mat A,Vec,Vec);
EXTERN int MatMultTransposeAdd_SeqDense(Mat A,Vec,Vec,Vec);

#endif
