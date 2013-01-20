/*
    Context for using preconditioned CG to minimize a quadratic function
 */

#if !defined(__QCG)
#define __QCG

typedef struct {
  PetscReal quadratic;
  PetscReal ltsnrm;
  PetscReal delta;
} KSP_QCG;

#endif
