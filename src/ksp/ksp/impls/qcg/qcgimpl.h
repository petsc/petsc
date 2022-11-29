/*
    Context for using preconditioned CG to minimize a quadratic function
 */

#ifndef PETSC_QCGIMPL_H
#define PETSC_QCGIMPL_H

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal quadratic;
  PetscReal ltsnrm;
  PetscReal delta;
} KSP_QCG;

#endif // PETSC_QCGIMPL_H
