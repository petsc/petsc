#if !defined(_FORESTIMPL_H)
#define _FORESTIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmforest.h> /*I      "petscdmforest.h"    I*/
#include <petscbt.h>
#include <petsc/private/dmimpl.h>
#include <../src/sys/utils/hash.h>

typedef struct {
  PetscInt                   refct;
  void                       *data;
  PetscErrorCode             (*createcellchart)(DM,PetscInt*,PetscInt*);
  PetscErrorCode             (*createcellsf)(DM,PetscSF*);
  PetscErrorCode             (*destroy)(DM);
  PetscErrorCode             (*ftemplate)(DM,DM);
  PetscErrorCode             (*mapcoordinates)(DM,PetscInt,PetscInt,const PetscReal[],PetscReal[],void*);
  void                       *mapcoordinatesctx;
  PetscBool                  setFromOptions;
  DMForestTopology           topology;
  DM                         base;
  DM                         adapt;
  PetscInt                   adjDim;
  PetscInt                   overlap;
  PetscInt                   minRefinement;
  PetscInt                   maxRefinement;
  PetscInt                   initRefinement;
  PetscInt                   cStart;
  PetscInt                   cEnd;
  PetscSF                    cellSF;
  char                       *adaptLabel;
  DMForestAdaptivityStrategy adaptStrategy;
  PetscInt                   gradeFactor;
  PetscReal                  *cellWeights;
  PetscCopyMode              cellWeightsCopyMode;
  PetscReal                  weightsFactor;
  PetscReal                  weightCapacity;
} DM_Forest;

PETSC_EXTERN PetscErrorCode DMCreate_Forest(DM);
PETSC_EXTERN PetscErrorCode DMClone_Forest(DM,DM*);
PETSC_EXTERN PetscErrorCode DMSetFromOptions_Forest(PetscOptionItems*,DM);

#endif /* _FORESTIMPL_H */
