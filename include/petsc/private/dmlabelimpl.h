#if !defined(_LABELIMPL_H)
#define _LABELIMPL_H

#include <petscdmlabel.h>
#include <petscbt.h>
#include <../src/sys/utils/hash.h>

/* This is an integer map, in addition it is also a container class
   Design points:
     - Low storage is the most important design point
     - We want flexible insertion and deletion
     - We can live with O(log) query, but we need O(1) iteration over strata
*/
struct _n_DMLabel {
  PetscInt         refct;
  PetscObjectState state;
  char       *name;           /* Label name */
  PetscInt    numStrata;      /* Number of integer values */
  PetscInt    defaultValue;   /* Background value when no value explicitly given */
  PetscInt   *stratumValues;  /* Value of each stratum */
  /* Basic sorted array storage */
  PetscBool  *arrayValid;     /* The array storage is valid (no additions need to be merged in) */
  PetscInt   *stratumSizes;   /* Size of each stratum */
  PetscInt  **points;         /* Points for each stratum, always sorted */
  /* Hashtable for fast insertion */
  PetscHashI *ht;             /* Hash table for fast insertion */
  /* Index for fast search */
  PetscInt    pStart, pEnd;   /* Bounds for index lookup */
  PetscBT     bt;             /* A bit-wise index */
};

#endif
