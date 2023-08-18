#pragma once

#include <petscmat.h>     /*I      "petscmat.h"        I*/
#include <petscdmpatch.h> /*I      "petscdmpatch.h"    I*/
#include <petsc/private/dmimpl.h>

typedef struct {
  PetscInt   refct;
  DM         dmCoarse;
  MatStencil patchSize;
  MatStencil commSize;
} DM_Patch;
