/*
 Context for limited memory variable metric method for unconstrained
 optimization.
*/

#pragma once
#include <petsc/private/taoimpl.h>

typedef struct {
  Mat M;

  Vec X;
  Vec G;
  Vec D;
  Vec W;
  Vec GV; /* the pseudo gradient */

  Vec Xold;
  Vec Gold;

  PetscInt bfgs;
  PetscInt sgrad;
  PetscInt grad;

  PetscReal lambda;
} TAO_OWLQN;

static PetscErrorCode ProjDirect_OWLQN(Vec d, Vec g);

static PetscErrorCode ComputePseudoGrad_OWLQN(Vec x, Vec gv, PetscReal lambda);
