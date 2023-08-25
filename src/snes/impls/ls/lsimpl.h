/*
   Private context for a Newton line search method for solving
   systems of nonlinear equations
 */

#pragma once
#include <petsc/private/snesimpl.h>

typedef struct {
  PetscInt dummy;
} SNES_NEWTONLS;
