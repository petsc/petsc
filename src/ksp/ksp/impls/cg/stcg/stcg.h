/*
    Context for using preconditioned CG to minimize a quadratic function
 */

#ifndef __STCG
#define __STCG

typedef struct {
  PetscReal quadratic;
  PetscReal radius;
} KSP_STCG;

#endif

