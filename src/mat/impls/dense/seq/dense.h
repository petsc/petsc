/* $Id: aij.h,v 1.15 1995/10/17 21:41:57 bsmith Exp $ */

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
} Mat_SeqDense;

#endif
