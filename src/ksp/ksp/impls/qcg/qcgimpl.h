/*
    Context for using preconditioned CG to minimize a quadratic function
 */

#pragma once

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal quadratic;
  PetscReal ltsnrm;
  PetscReal delta;
} KSP_QCG;
