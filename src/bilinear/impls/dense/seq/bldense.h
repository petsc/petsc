/* $Id: dense.h,v 1.1 1999/03/27 22:10:10 knepley Exp $ */

#include "src/bilinear/bilinearimpl.h"

#if !defined(__BILINEAR_DENSE_H)
#define __BILINEAR_DENSE_H

/*  
  BILINEAR_DENSE_SEQ format - conventional dense C storage (by rows)
*/

typedef struct {
  Scalar    *v;             /* The operator elements */
  PetscTruth user_alloc;    /* The flag indicating the user provided the dense data */
  int       *pivots;        /* The pivots in LU factorization */
} Bilinear_Dense_Seq;

extern int MatMult_SeqDense(Mat A,Vec,Vec);

#endif
