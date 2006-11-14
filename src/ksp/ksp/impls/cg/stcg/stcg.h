/*
    Context for using preconditioned CG to minimize a quadratic function
 */

#ifndef __STCG
#define __STCG

#define STCG_PRECONDITIONED_DIRECTION 0
#define STCG_UNPRECONDITIONED_DIRECTION 1
#define STCG_DIRECTION_TYPES 2

typedef struct {
  PetscReal radius;
  PetscReal norm_d;
  int dtype;
} KSP_STCG;

#endif

