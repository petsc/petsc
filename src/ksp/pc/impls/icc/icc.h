/* $Id: icc.h,v 1.11 2001/08/07 21:30:30 bsmith Exp $ */
#include "src/ksp/pc/pcimpl.h"          

#if !defined(__ICC_H)
#define __ICC_H

/* Incomplete Cholesky factorization context */

typedef struct {
  Mat             fact;
  MatOrderingType ordering;
  MatFactorInfo   info;
  void            *implctx;
} PC_ICC;

#endif
