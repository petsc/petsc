#ifndef PETSCFETYPES_H
#define PETSCFETYPES_H

/* SUBMANSEC = FE */

/*S
  PetscFE - PETSc object that manages a finite element space, e.g. the P_1 Lagrange element

  Level: beginner

.seealso: `PetscFECreate()`, `PetscSpace`, `PetscDualSpace`, `PetscSpaceCreate()`, `PetscDualSpaceCreate()`, `PetscFESetType()`, `PetscFEType`
S*/
typedef struct _p_PetscFE *PetscFE;

/*MC
  PetscFEJacobianType - indicates which pointwise functions should be used to fill the Jacobian matrix

  Level: beginner

.seealso: `PetscFEIntegrateJacobian()`
M*/
typedef enum {
  PETSCFE_JACOBIAN,
  PETSCFE_JACOBIAN_PRE,
  PETSCFE_JACOBIAN_DYN
} PetscFEJacobianType;

#endif
