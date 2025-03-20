#pragma once

/* MANSEC = DM */
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

/*E
  PetscFEGeomMode - Describes the type of geometry being encoded.

  Values:
+ `PETSC_FEGEOM_BASIC`    - These are normal dim-cells, with dim == dE, and only bulk data is stored.
. `PETSC_FEGEOM_EMBEDDED` - These are dim-cells embedded in a higher dimension, as an embedded manifold, where dim < dE and only bulk data is stored.
. `PETSC_FEGEOM_BOUNDARY` - These are dim-cells on the boundary of a dE-mesh, so that dim < dE, and both bulk and s = 1 face data are stored.
- `PETSC_FEGEOM_COHESIVE` - These are dim-cells in the interior of a dE-mesh, so that dim < dE, and both bulk and s = 2 face data are stored.

  Level: beginner

  Note:
  .vb
  dim - The topological dimension and reference coordinate dimension
  dE  - The real coordinate dimension
  s   - The number of supporting cells for a face
  .ve

.seealso: [](ch_dmbase), `PetscFEGeom`, `DM`, `DMPLEX`, `PetscFEGeomCreate()`
E*/
typedef enum {
  PETSC_FEGEOM_BASIC,
  PETSC_FEGEOM_EMBEDDED,
  PETSC_FEGEOM_BOUNDARY,
  PETSC_FEGEOM_COHESIVE
} PetscFEGeomMode;
