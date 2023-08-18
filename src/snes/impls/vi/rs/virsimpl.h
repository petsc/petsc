#pragma once

/*
   Private context for reduced space active set newton method with line search for solving
   system of mixed complementarity equations
 */

#include <petsc/private/snesimpl.h>

typedef struct {
  PetscErrorCode (*checkredundancy)(SNES, IS, IS *, void *);

  void *ctxP; /* user defined check redundancy context */
  IS    IS_inact_prev;
  IS    IS_inact;
} SNES_VINEWTONRSLS;
