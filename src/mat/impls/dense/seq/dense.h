/* $Id: dense.h,v 1.2 1995/11/21 22:23:55 curfman Exp curfman $ */

#include "matimpl.h"
#include <math.h>

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
