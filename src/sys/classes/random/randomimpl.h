
#ifndef __PETSCRANDOMIMPL_H
#define __PETSCRANDOMIMPL_H

#include <petscsys.h>

typedef struct _PetscRandomOps *PetscRandomOps;
struct _PetscRandomOps {
  /* 0 */
  PetscErrorCode  (*seed)(PetscRandom);
  PetscErrorCode  (*getvalue)(PetscRandom,PetscScalar*);
  PetscErrorCode  (*getvaluereal)(PetscRandom,PetscReal*);
  PetscErrorCode  (*destroy)(PetscRandom);
  PetscErrorCode  (*setfromoptions)(PetscRandom);
};

struct _p_PetscRandom {
  PETSCHEADER(struct _PetscRandomOps);
  void        *data;           /* implementation-specific data */
  unsigned    long seed;
  PetscScalar low,width;       /* lower bound and width of the interval over
                                  which the random numbers are distributed */
  PetscBool   iset;            /* if true, indicates that the user has set the interval */
  /* array for shuffling ??? */
};

#endif

