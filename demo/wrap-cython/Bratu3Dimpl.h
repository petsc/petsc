#ifndef BRATU3D_H
#define BRATU3D_H

#include <petsc.h>
#include <petscvec.h>
#include <petscmat.h>
#include <petscda.h>

typedef struct Params {
  double lambda_;
} Params;

PetscErrorCode FormInitGuess(DA da, Vec x, Params *p);
PetscErrorCode FormFunction(DA da, Vec x, Vec F, Params *p);
PetscErrorCode FormJacobian(DA da, Vec x, Mat J, Params *p);

#endif /* !BRATU3D_H */
