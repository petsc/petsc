/*
   This private file should not be included in users' code.
*/

#pragma once

#include <petscao.h>
#include <petsc/private/petscimpl.h>
#include <petscviewer.h>

PETSC_INTERN PetscFunctionList AOList;

/*
    Defines the abstract AO operations
*/
typedef struct _AOOps *AOOps;
struct _AOOps {
  /* Generic Operations */
  PetscErrorCode (*view)(AO, PetscViewer);
  PetscErrorCode (*destroy)(AO);
  /* AO-Specific Operations */
  PetscErrorCode (*petsctoapplication)(AO, PetscInt, PetscInt[]);
  PetscErrorCode (*applicationtopetsc)(AO, PetscInt, PetscInt[]);
  PetscErrorCode (*petsctoapplicationpermuteint)(AO, PetscInt, PetscInt[]);
  PetscErrorCode (*applicationtopetscpermuteint)(AO, PetscInt, PetscInt[]);
  PetscErrorCode (*petsctoapplicationpermutereal)(AO, PetscInt, PetscReal[]);
  PetscErrorCode (*applicationtopetscpermutereal)(AO, PetscInt, PetscReal[]);
};

struct _p_AO {
  PETSCHEADER(struct _AOOps);
  PetscInt N, n;    /* global, local ao size */
  IS       isapp;   /* index set that defines an application ordering provided by user */
  IS       ispetsc; /* index set that defines PETSc ordering provided by user */
  void    *data;    /* implementation-specific data */
};

extern PetscLogEvent AO_PetscToApplication, AO_ApplicationToPetsc;
