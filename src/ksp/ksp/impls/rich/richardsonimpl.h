/*
      Private data structure for Richardson Iteration
*/

#ifndef __RICHARDSONIMPL_H
#define __RICHARDSONIMPL_H

#include <petsc/private/kspimpl.h>

typedef struct {
  PetscReal scale;     /* scaling on preconditioner */
  PetscBool selfscale; /* determine optimimal scaling each iteration to minimize 2-norm of resulting residual */
} KSP_Richardson;

#endif
