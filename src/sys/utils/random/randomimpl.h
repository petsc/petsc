
#ifndef __PETSCRANDOMIMPL_H
#define __PETSCRANDOMIMPL_H

#include "petscsys.h"

typedef struct _PetscRandomOps *PetscRandomOps;
struct _PetscRandomOps {
  PetscErrorCode PETSC_DLLEXPORT (*seed)(PetscRandom);
  PetscErrorCode PETSC_DLLEXPORT (*getvalues)(PetscRandom,PetscScalar*);
};

struct _p_PetscRandom {
  // PETSCHEADER(int);
  PETSCHEADER(struct _PetscRandomOps);
  void        *data;           /* implementation-specific data */
  unsigned    long seed;
  PetscScalar low,width;       /* lower bound and width of the interval over
                                  which the random numbers are distributed */
  PetscTruth  iset;            /* if true, indicates that the user has set the interval */
  /* array for shuffling ??? */
};

#endif

