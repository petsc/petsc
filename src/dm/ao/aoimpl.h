/* 
   This private file should not be included in users' code.
*/

#ifndef __AOIMPL 
#define __AOIMPL

#include "petscao.h"

/*
    Defines the abstract AO operations
*/
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
  void          *data;                   /* implementation-specific data */
  PetscInt      N,n;                    /* global, local vector size */
};

extern PetscLogEvent  AO_PetscToApplication, AO_ApplicationToPetsc;


#endif
