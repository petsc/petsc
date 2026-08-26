/*
      Private data structure for Richardson Iteration
*/

#pragma once

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal        scale;     /* scaling on preconditioner */
  PetscBool        selfscale; /* determine optimimal scaling each iteration to minimize 2-norm of resulting residual */
  Mat              R, Z;      /* work blocks of vectors of KSPMatSolve_Richardson(), cached between calls, R also holds the product with the operator */
  Vec              w;         /* work vector of the PCApplyRichardson() fallback of KSPMatSolve_Richardson(), cached between calls */
  PetscBool        transpose; /* direction the product cached in R was set up with */
  PetscObjectId    id;        /* identity of the operator the product cached in R was set up with */
  PetscObjectState state;     /* nonzero state of that operator, only a change of nonzero pattern invalidates the symbolic phase, new values do not */
} KSP_Richardson;
