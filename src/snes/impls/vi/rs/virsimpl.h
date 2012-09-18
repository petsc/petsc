#include <petsc-private/snesimpl.h>

/*
   Private context for reduced space active set newton method with line search for solving
   system of mixed complementarity equations
 */

#ifndef __SNES_VIRS_H
#define __SNES_VIRS_H

typedef struct {
  PetscErrorCode (*checkredundancy)(SNES,IS,IS*,void*);
  void           *ctxP; /* user defined check redundancy context */
  IS             IS_inact_prev;
} SNES_VIRS;

#endif

