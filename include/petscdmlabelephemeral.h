#pragma once

#include <petscdmlabel.h>
#include <petscdmplextransform.h>

/* MANSEC = DM */

PETSC_EXTERN PetscErrorCode DMLabelEphemeralGetLabel(DMLabel, DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelEphemeralSetLabel(DMLabel, DMLabel);
PETSC_EXTERN PetscErrorCode DMLabelEphemeralGetTransform(DMLabel, DMPlexTransform *);
PETSC_EXTERN PetscErrorCode DMLabelEphemeralSetTransform(DMLabel, DMPlexTransform);
