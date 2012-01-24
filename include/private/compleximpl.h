#if !defined(_COMPLEXIMPL_H)
#define _COMPLEXIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmcomplex.h> /*I      "petscdmcomplex.h"    I*/
#include "private/dmimpl.h"

typedef struct Sieve_Label *SieveLabel;
struct Sieve_Label {
  char      *name;           /* Label name */
  PetscInt   numStrata;      /* Number of integer values */
  PetscInt  *stratumValues;  /* Value of each stratum */
  PetscInt  *stratumOffsets; /* Offset of each stratum */
  PetscInt  *stratumSizes;   /* Size of each stratum */
  PetscInt  *points;         /* Points for each stratum, sorted after setup */
  SieveLabel next;           /* Linked list */
};

typedef struct {
  PetscInt             dim; /* Topological mesh dimension */
  PetscSF              sf;  /* SF for parallel point overlap */

  /*   Sieve */
  PetscSection         coneSection;    /* Layout of cones (inedges for DAG) */
  PetscInt             maxConeSize;    /* Cached for fast lookup */
  PetscInt            *cones;          /* Cone for each point */
  PetscInt            *coneOrientations; /* TODO */
  PetscSection         supportSection; /* Layout of cones (inedges for DAG) */
  PetscInt             maxSupportSize; /* Cached for fast lookup */
  PetscInt            *supports;       /* Cone for each point */
  PetscSection         coordSection;   /* Layout for coordinates */
  Vec                  coordinates;    /* Coordinate values */

  PetscInt            *meetTmpA,    *meetTmpB;    /* Work space for meet operation */
  PetscInt            *joinTmpA,    *joinTmpB;    /* Work space for join operation */
  PetscInt            *closureTmpA, *closureTmpB; /* Work space for closure operation */

  /* Labels */
  SieveLabel           labels;         /* Linked list of labels */

  PetscSection         defaultSection;
} DM_Complex;

#endif /* _COMPLEXIMPL_H */
