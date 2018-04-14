#ifndef __TAO_PGD_H
#define __TAO_PGD_H

#include <petsc/private/taoimpl.h>

typedef struct {
  Vec unprojected_gradient;
  PetscReal stepsize;
  PetscReal f;
  PetscReal gnorm;
} TAO_PGD;

#endif

