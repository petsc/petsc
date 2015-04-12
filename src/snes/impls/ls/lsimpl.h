/*
   Private context for a Newton line search method for solving
   systems of nonlinear equations
 */

#if !defined(__SNES_LS_H)
#define __SNES_LS_H
#include <petsc/private/snesimpl.h>

typedef struct {
  PetscInt dummy;
} SNES_NEWTONLS;

#endif
