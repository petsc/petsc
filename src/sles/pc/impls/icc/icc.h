/* $Id: icc.h,v 1.10 2000/09/22 20:45:06 bsmith Exp bsmith $ */
#include "src/sles/pc/pcimpl.h"          

#if !defined(__ICC_H)
#define __ICC_H

/* Incomplete Cholesky factorization context */

typedef struct {
  Mat             fact;
  MatOrderingType ordering;
  int             levels;
  PetscReal       fill;
  void            *implctx;
} PC_ICC;

#endif
