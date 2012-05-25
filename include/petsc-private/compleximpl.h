#if !defined(_COMPLEXIMPL_H)
#define _COMPLEXIMPL_H

#include <petscmat.h>       /*I      "petscmat.h"          I*/
#include <petscdmcomplex.h> /*I      "petscdmcomplex.h"    I*/
#include "petsc-private/dmimpl.h"

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
  PetscInt             dim;   /* Topological mesh dimension */

  /* Sieve */
  PetscSection         coneSection;      /* Layout of cones (inedges for DAG) */
  PetscInt             maxConeSize;      /* Cached for fast lookup */
  PetscInt            *cones;            /* Cone for each point */
  PetscInt            *coneOrientations; /* Orientation of each cone point, means cone traveral should start on point 'o', and if negative start on -(o+1) and go in reverse */
  PetscSection         supportSection;   /* Layout of cones (inedges for DAG) */
  PetscInt             maxSupportSize;   /* Cached for fast lookup */
  PetscInt            *supports;         /* Cone for each point */
  PetscSection         coordSection;     /* Layout for coordinates */
  Vec                  coordinates;      /* Coordinate values */
  PetscReal            refinementLimit;  /* Maximum volume for refined cell */

  PetscInt            *meetTmpA,    *meetTmpB;    /* Work space for meet operation */
  PetscInt            *joinTmpA,    *joinTmpB;    /* Work space for join operation */
  PetscInt            *closureTmpA, *closureTmpB; /* Work space for closure operation */
  PetscInt            *facesTmp;                  /* Work space for faces operation */

  /* Labels */
  SieveLabel           labels;         /* Linked list of labels */

  /* Debugging */
  PetscBool               printSetValues;
} DM_Complex;


#endif /* _COMPLEXIMPL_H */
