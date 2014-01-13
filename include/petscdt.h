/*
  Common tools for constructing discretizations
*/
#if !defined(__PETSCDT_H)
#define __PETSCDT_H

#include <petscsys.h>

typedef struct {
  PetscInt         dim;       /* The spatial dimension */
  PetscInt         numPoints; /* The number of quadrature points on an element */
  const PetscReal *points;    /* The quadrature point coordinates */
  const PetscReal *weights;   /* The quadrature weights */
} PetscQuadrature;

typedef struct {
  PetscReal *v0, *n, *J, *invJ, *detJ;
} PetscCellGeometry;

PETSC_EXTERN PetscErrorCode PetscDTLegendreEval(PetscInt,const PetscReal*,PetscInt,const PetscInt*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussQuadrature(PetscInt,PetscReal,PetscReal,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTReconstructPoly(PetscInt,PetscInt,const PetscReal*,PetscInt,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussJacobiQuadrature(PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);
PETSC_EXTERN PetscErrorCode PetscQuadratureView(PetscQuadrature,PetscViewer);
PETSC_EXTERN PetscErrorCode PetscQuadratureDestroy(PetscQuadrature*);

#endif
