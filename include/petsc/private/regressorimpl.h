#pragma once

#include <petscregressor.h>
#include <petsc/private/petscimpl.h>

PETSC_EXTERN PetscBool      PetscRegressorRegisterAllCalled;
PETSC_EXTERN PetscErrorCode PetscRegressorRegisterAll(void);

typedef struct _PetscRegressorOps *PetscRegressorOps;

struct _PetscRegressorOps {
  PetscErrorCode (*setup)(PetscRegressor);
  PetscErrorCode (*setfromoptions)(PetscRegressor, PetscOptionItems); /* sets options from database */
  PetscErrorCode (*settraining)(PetscRegressor, Mat, Vec);            /* set the training data matrix and targets */
  PetscErrorCode (*fit)(PetscRegressor);                              /* compute the transformation to be applied */
  PetscErrorCode (*predict)(PetscRegressor, Mat, Vec);                /* predict using fitted model */
  PetscErrorCode (*destroy)(PetscRegressor);
  PetscErrorCode (*reset)(PetscRegressor);
  PetscErrorCode (*view)(PetscRegressor, PetscViewer);
};

/* Define the PetscRegressor data structure. */
struct _p_PetscRegressor {
  PETSCHEADER(struct _PetscRegressorOps);

  PetscBool setupcalled; /* True if setup has been called */
  PetscBool fitcalled;   /* True if the Fit() method has been called. */
  void     *data;        /* Implementation-specific data */
  Mat       training;    /* Matrix holding the training data set */
  Vec       target;      /* Targets for training data (response variables or labels) */
  Tao       tao;         /* Tao optimizer used by many regressor implementations */
  PetscObjectParameterDeclare(PetscReal, regularizer_weight);
};

PETSC_EXTERN PetscLogEvent PetscRegressor_SetUp;
PETSC_EXTERN PetscLogEvent PetscRegressor_Fit;
PETSC_EXTERN PetscLogEvent PetscRegressor_Predict;
