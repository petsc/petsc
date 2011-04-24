#ifndef BRATU3D_H
#define BRATU3D_H

#include <petsc.h>

#if PETSC_VERSION_(3,1,0)
#include <petscvec.h>
#include <petscmat.h>
#include <petscda.h>
#endif

typedef struct Params {
  double lambda_;
} Params;

PetscErrorCode FormInitGuess(DM da, Vec x, Params *p);
PetscErrorCode FormFunction(DM da, Vec x, Vec F, Params *p);
PetscErrorCode FormJacobian(DM da, Vec x, Mat J, Params *p);

#endif /* !BRATU3D_H */
