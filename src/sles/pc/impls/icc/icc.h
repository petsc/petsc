/* static char vcid[] = "$Id: icc.h,v 1.6 1999/01/27 19:46:54 bsmith Exp bsmith $ "; */
#include "src/sles/pc/pcimpl.h"          

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
