#if !defined(_FORESTIMPL_H)
#define _FORESTIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmforest.h> /*I      "petscdmforest.h"    I*/
#include <petscbt.h>
#include <petsc-private/dmimpl.h>
#include <../src/sys/utils/hash.h>

typedef struct {
  PetscInt                   refct;
  void                       *data;
  PetscBool                  setup;
  DMForestType               type;
  DMForestTopology           topology;
  DM                         base;
  DM                         coarse;
  DM                         fine;
  PetscInt                   adjDim;
  PestcInt                   overlap;
  PetscInt                   minRefinement;
  PetscInt                   maxRefinement;
  PetscInt                   cStart;
  PetscInt                   cEnd;
  PetscSF                    cellSF;
  PetscInt                   *adaptMarkers;
  PetscCopyMode              adaptCopyMode;
  DMForestAdaptivityStrategy adaptStrategy;
  PetscInt                   gradeFactor;
  PestcReal                  *cellWeights;
  PetscCopyMode              cellWeightsCopyMode;
  PetscReal                  weightsFactor;
  PetscReal                  weightCapacity;
} DM_Forest;

#endif /* _FORESTIMPL_H */
