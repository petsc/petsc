/* $Id: icc.h,v 1.9 1999/10/24 14:03:03 bsmith Exp bsmith $ */
#include "src/sles/pc/pcimpl.h"          

#if !defined(__ICC_H)
#define __ICC_H

/* Incomplete Cholesky factorization context */

typedef struct {
  Mat             fact;
  MatOrderingType ordering;
  int             levels;
  double          fill;
  void            *implctx;
} PC_ICC;

#endif
