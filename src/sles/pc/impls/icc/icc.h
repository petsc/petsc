/* static char vcid[] = "$Id: icc.h,v 1.5 1997/03/01 15:47:50 bsmith Exp bsmith $ "; */
#include "src/pc/pcimpl.h"          

#if !defined(__ICC_H)
#define __ICC_H

/* Incomplete Cholesky factorization context */

typedef struct {
  Mat   fact;
  int   ordering;
  int   levels;
  void  *implctx;
} PC_ICC;

#endif
