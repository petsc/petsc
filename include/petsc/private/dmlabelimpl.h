#ifndef _LABELIMPL_H
#define _LABELIMPL_H

#include <petscdmlabel.h>
#include <petscbt.h>
#include <petscistypes.h>
#include <petsc/private/hashmapi.h>
#include <petsc/private/hashseti.h>

typedef struct _p_DMLabelOps *DMLabelOps;
struct _p_DMLabelOps {
  PetscErrorCode (*view)(DMLabel, PetscViewer);
  PetscErrorCode (*setup)(DMLabel);
  PetscErrorCode (*destroy)(DMLabel);
  PetscErrorCode (*duplicate)(DMLabel, DMLabel *);
  PetscErrorCode (*getstratumis)(DMLabel, PetscInt, IS *);
};

/* This is an integer map, in addition it is also a container class
   Design points:
     - Low storage is the most important design point
     - We want flexible insertion and deletion
     - We can live with O(log) query, but we need O(1) iteration over strata
*/
struct _p_DMLabel {
  PETSCHEADER(struct _p_DMLabelOps);
  PetscBool readonly;      /* Flag for labels which cannot be modified after creation */
  PetscInt  numStrata;     /* Number of integer values */
  PetscInt  defaultValue;  /* Background value when no value explicitly given */
  PetscInt *stratumValues; /* Value of each stratum */
  /* Basic IS storage */
  PetscBool *validIS;      /* The IS is valid (no additions need to be merged in) */
  PetscInt  *stratumSizes; /* Size of each stratum */
  IS        *points;       /* Points for each stratum, always sorted */
  /* Hash tables for fast search and insertion */
  PetscHMapI  hmap; /* Hash map for fast strata search */
  PetscHSetI *ht;   /* Hash set for fast insertion */
  /* Index for fast search */
  PetscInt pStart, pEnd; /* Bounds for index lookup */
  PetscBT  bt;           /* A bit-wise index */
  /* Propagation */
  PetscInt *propArray; /* Array of values for propagation */
};

PETSC_INTERN PetscErrorCode DMLabelLookupStratum(DMLabel, PetscInt, PetscInt *);
PETSC_INTERN PetscErrorCode DMLabelGetStratumSize_Private(DMLabel, PetscInt, PetscInt *);
PETSC_INTERN PetscErrorCode PetscSectionSymCreate_Label(PetscSectionSym);
PETSC_INTERN PetscErrorCode DMLabelMakeAllInvalid_Internal(DMLabel);
#endif
