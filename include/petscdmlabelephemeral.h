#ifndef PETSCDMLABEL_EPH_H
#define PETSCDMLABEL_EPH_H
#include <petscdmlabel.h>
#include <petscdmplextransform.h>

PETSC_EXTERN PetscErrorCode DMLabelEphemeralGetLabel(DMLabel, DMLabel *);
PETSC_EXTERN PetscErrorCode DMLabelEphemeralSetLabel(DMLabel, DMLabel);
PETSC_EXTERN PetscErrorCode DMLabelEphemeralGetTransform(DMLabel, DMPlexTransform *);
PETSC_EXTERN PetscErrorCode DMLabelEphemeralSetTransform(DMLabel, DMPlexTransform);

#endif
