/* $Id: dense.h,v 1.4 1996/08/08 14:42:39 bsmith Exp bsmith $ */

#include "src/mat/matimpl.h"

#if !defined(__DENSE_H)
#define __DENSE_H

/*  
  MATSEQDENSE format - conventional dense Fortran storage (by columns)
*/

typedef struct {
  Scalar *v;                /* matrix elements */
  int    roworiented;       /* if true, row oriented input (default) */
  int    m, n;              /* rows, columns */
  int    pad;               /* padding */        
  int    *pivots;           /* pivots in LU factorization */
  int    user_alloc;        /* true if the user provided the dense data */
} Mat_SeqDense;

extern int MatMult_SeqDense(Mat A,Vec,Vec);
extern int MatMultAdd_SeqDense(Mat A,Vec,Vec,Vec);
extern int MatMultTrans_SeqDense(Mat A,Vec,Vec);
extern int MatMultTransAdd_SeqDense(Mat A,Vec,Vec,Vec);

#endif
