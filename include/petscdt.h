/*
  Common tools for constructing discretizations
*/
#if !defined(PETSCDT_H)
#define PETSCDT_H

#include <petscsys.h>

/*S
  PetscQuadrature - Quadrature rule for integration.

  Level: beginner

.seealso:  PetscQuadratureCreate(), PetscQuadratureDestroy()
S*/
typedef struct _p_PetscQuadrature *PetscQuadrature;

/*E
  PetscGaussLobattoLegendreCreateType - algorithm used to compute the Gauss-Lobatto-Legendre nodes and weights

  Level: intermediate

$  PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA - compute the nodes via linear algebra
$  PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON - compute the nodes by solving a nonlinear equation with Newton's method

E*/
typedef enum {PETSCGAUSSLOBATTOLEGENDRE_VIA_LINEAR_ALGEBRA,PETSCGAUSSLOBATTOLEGENDRE_VIA_NEWTON} PetscGaussLobattoLegendreCreateType;

PETSC_EXTERN PetscErrorCode PetscQuadratureCreate(MPI_Comm, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscQuadratureDuplicate(PetscQuadrature, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetOrder(PetscQuadrature, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetOrder(PetscQuadrature, PetscInt);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetNumComponents(PetscQuadrature, PetscInt*);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetNumComponents(PetscQuadrature, PetscInt);
PETSC_EXTERN PetscErrorCode PetscQuadratureGetData(PetscQuadrature, PetscInt*, PetscInt*, PetscInt*, const PetscReal *[], const PetscReal *[]);
PETSC_EXTERN PetscErrorCode PetscQuadratureSetData(PetscQuadrature, PetscInt, PetscInt, PetscInt, const PetscReal [], const PetscReal []);
PETSC_EXTERN PetscErrorCode PetscQuadratureView(PetscQuadrature, PetscViewer);
PETSC_EXTERN PetscErrorCode PetscQuadratureDestroy(PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscQuadratureExpandComposite(PetscQuadrature, PetscInt, const PetscReal[], const PetscReal[], PetscQuadrature *);

PETSC_EXTERN PetscErrorCode PetscDTLegendreEval(PetscInt,const PetscReal*,PetscInt,const PetscInt*,PetscReal*,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussQuadrature(PetscInt,PetscReal,PetscReal,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussLobattoLegendreQuadrature(PetscInt,PetscGaussLobattoLegendreCreateType,PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTReconstructPoly(PetscInt,PetscInt,const PetscReal*,PetscInt,const PetscReal*,PetscReal*);
PETSC_EXTERN PetscErrorCode PetscDTGaussTensorQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);
PETSC_EXTERN PetscErrorCode PetscDTGaussJacobiQuadrature(PetscInt,PetscInt,PetscInt,PetscReal,PetscReal,PetscQuadrature*);

PETSC_EXTERN PetscErrorCode PetscDTTanhSinhTensorQuadrature(PetscInt, PetscInt, PetscReal, PetscReal, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode PetscDTTanhSinhIntegrate(void (*)(PetscReal, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscDTTanhSinhIntegrateMPFR(void (*)(PetscReal, PetscReal *), PetscReal, PetscReal, PetscInt, PetscReal *);

PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreIntegrate(PetscInt, PetscReal *, PetscReal *, const PetscReal *, PetscReal *);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementLaplacianCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementLaplacianDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementGradientCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementGradientDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementAdvectionCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementAdvectionDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementMassCreate(PetscInt, PetscReal *, PetscReal *, PetscReal ***);
PETSC_EXTERN PetscErrorCode PetscGaussLobattoLegendreElementMassDestroy(PetscInt, PetscReal *, PetscReal *, PetscReal ***);

#endif
