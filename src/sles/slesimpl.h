/* $Id: slesimpl.h,v 1.11 1999/11/23 18:08:52 bsmith Exp balay $ */

#if !defined(_SLESIMPL_H)
#define _SLESIMPL_H
#include "petscsles.h"

struct _p_SLES {
  PETSCHEADER(int)
  int        setupcalled;
  PetscTruth dscale;      /* diagonal scale system; used with SLESSetDiagonalScale() */
  PetscTruth dscalefix;   /* unscale system after solve */
  PetscTruth dscalefix2;  /* system has been unscaled */
  Vec        diagonal;    /* 1/sqrt(diag of matrix) */
  PC         pc;
  KSP        ksp;
};

#endif
