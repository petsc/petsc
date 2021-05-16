#ifndef __TAO_GPCGLINESEARCH_H
#define __TAO_GPCGLINESEARCH_H

#include <petscvec.h>

typedef struct {

/* --------------- Parameters used by line search method ----------------- */
  PetscReal maxstep;         /* maximum step size */
  PetscInt    bracket;
  PetscInt    infoc;

  Vec x;
  Vec W1;
  Vec W2;
  Vec Gold;

} TaoLineSearch_GPCG;

#endif
