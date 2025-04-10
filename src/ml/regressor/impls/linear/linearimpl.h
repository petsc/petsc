#pragma once
#include <petsc/private/regressorimpl.h>
#include <petscksp.h>
#include <petsctao.h>

/* We define this header, since it serves as a "base" for all linear models. */
#define REGRESSOR_LINEAR_HEADER \
  PetscRegressorLinearType type; \
  /* Parameters of the fitted regression model */ \
  Vec         coefficients; \
  PetscScalar intercept; \
\
  Mat X;        /* Operator of the linear model; often the training data matrix, but might be a MATCOMPOSITE */ \
  Mat C;        /* Centering matrix */ \
  Vec rhs;      /* Right-hand side of the linear model; often the target vector, but may be the mean-centered version */ \
  Vec residual; /* Residual for our model, or the loss vector */ \
  /* Various options */ \
  PetscBool fit_intercept; /* Calculate intercept ("bias" or "offset") if true. Assume centered data if false. */ \
  PetscBool use_ksp        /* Use KSP for the model-fitting problem; otherwise we will use TAO. */

typedef struct {
  REGRESSOR_LINEAR_HEADER;

  PetscInt ksp_its, ksp_tot_its;
  KSP      ksp;
  Mat      XtX; /* Normal matrix formed from X */
} PetscRegressor_Linear;
