/*
    Context for using preconditioned CG to minimize a quadratic function 
 */

#ifndef __QCG
#define __QCG

typedef struct {
  PetscReal quadratic;
  PetscReal ltsnrm;
  PetscReal delta;
} KSP_QCG;

#endif
