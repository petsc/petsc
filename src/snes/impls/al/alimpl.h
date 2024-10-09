/*
   Private context for a Newton arc length continuation
   method for solving systems of nonlinear equations
*/

#pragma once
#include <petsc/private/snesimpl.h>
#include <petscsnes.h>

typedef struct {
  PetscInt                   max_continuation_steps; /* maximum number of continuation steps */
  PetscReal                  step_size;              /* radius of quadratic constraint surface */
  PetscReal                  psisq;                  /* load factor regularization parameter */
  PetscReal                  lambda_update;          /* accumulated update to lambda over the current increment */
  PetscReal                  lambda;                 /* load parameter */
  PetscReal                  lambda_max;             /* maximum value of the load parameter */
  PetscReal                  lambda_min;             /* minimum value of the load parameter */
  PetscBool                  scale_rhs;              /* should the RHS vector be scaled by the load parameter? */
  SNESNewtonALCorrectionType correction_type;        /* type of correction scheme to use */
  PetscBool                  copied_rhs;             /* has the right-hand side vector been copied? */

  Vec vec_rhs_orig; /* original right-hand side vector, used if `scale_rhs == PETSC_TRUE` */

  SNESFunctionFn *computealfunction; /* user-provided function to compute the tangent load vector  */
  void           *alctx;             /* user-provided context for the tangent load vector computation */
} SNES_NEWTONAL;

PETSC_INTERN const char NewtonALExactCitation[];
PETSC_INTERN PetscBool  NewtonALExactCitationSet;
PETSC_INTERN const char NewtonALNormalCitation[];
PETSC_INTERN PetscBool  NewtonALNormalCitationSet;
