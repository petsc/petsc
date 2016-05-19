/*
    Context for using preconditioned CG to minimize a quadratic function
 */

#if !defined(__QCG)
#define __QCG

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal quadratic;
  PetscReal ltsnrm;
  PetscReal delta;
} KSP_QCG;

#endif
