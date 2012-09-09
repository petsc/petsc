#if !defined(_PATCHIMPL_H)
#define _PATCHIMPL_H

#include <petscmat.h>     /*I      "petscmat.h"        I*/
#include <petscdmpatch.h> /*I      "petscdmpatch.h"    I*/
#include "petsc-private/dmimpl.h"

typedef struct {
  PetscInt refct;
  PetscInt dim;
  PetscInt numPatches;
  DM      *patches;
  DM       dmCoarse;
  PetscInt activePatch;
} DM_Patch;

#endif /* _PATCHIMPL_H */
