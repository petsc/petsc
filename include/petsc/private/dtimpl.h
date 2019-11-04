#if !defined(_DT_H)
#define _DT_H

#include <petscdt.h>

struct _p_PetscQuadrature {
  PETSCHEADER(int);
  PetscInt         dim;       /* The spatial dimension */
  PetscInt         Nc;        /* The number of components */
  PetscInt         order;     /* The order, i.e. the highest degree polynomial that is exactly integrated */
  PetscInt         numPoints; /* The number of quadrature points on an element */
  const PetscReal *points;    /* The quadrature point coordinates */
  const PetscReal *weights;   /* The quadrature weights */
};

PETSC_STATIC_INLINE PetscErrorCode PetscDTFactorial_Internal(PetscInt n, PetscReal *factorial)
{
  PetscReal f = 1.0;
  PetscInt  i;

  PetscFunctionBegin;
  for (i = 1; i < n+1; ++i) f *= i;
  *factorial = f;
  PetscFunctionReturn(0);
}

PETSC_STATIC_INLINE PetscErrorCode PetscDTFactorialInt_Internal(PetscInt n, PetscInt *factorial)
{
  PetscFunctionBeginHot;
  if (n <= 3) {
    PetscInt facLookup[4] = {1, 1, 2, 6};

    *factorial = facLookup[n];
  } else {
    PetscInt f = 1;
    PetscInt i;

    for (i = 1; i < n+1; ++i) f *= i;
    *factorial = f;
  }
  PetscFunctionReturn(0);
}

#endif
