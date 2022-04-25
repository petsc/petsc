#if !defined(PETSCFVTYPES_H)
#define PETSCFVTYPES_H

/*S
  PetscLimiter - PETSc object that manages a finite volume slope limiter

  Level: beginner

.seealso: `PetscLimiterCreate()`, `PetscLimiterSetType()`, `PetscLimiterType`
S*/
typedef struct _p_PetscLimiter *PetscLimiter;

/*S
  PetscFV - PETSc object that manages a finite volume discretization

  Level: beginner

.seealso: `PetscFVCreate()`, `PetscFVSetType()`, `PetscFVType`
S*/
typedef struct _p_PetscFV *PetscFV;

/*S
  PetscFVFaceGeom - Data structure (C struct) for storing information about face geometry for a finite volume method.

  Level: beginner

  Note: The components are
$  PetscReal   normal[3]   - Area-scaled normals
$  PetscReal   centroid[3] - Location of centroid (quadrature point)
$  PetscScalar grad[2][3]  - Face contribution to gradient in left and right cell

.seealso: `DMPlexComputeGeometryFVM()`
S*/
typedef struct {
  PetscReal   normal[3];   /* Area-scaled normals */
  PetscReal   centroid[3]; /* Location of centroid (quadrature point) */
  PetscScalar grad[2][3];  /* Face contribution to gradient in left and right cell */
} PetscFVFaceGeom;

/*S
  PetscFVCellGeom - Data structure (C struct) for storing information about cell geometry for a finite volume method.

  Level: beginner

  Note: The components are
$  PetscReal   centroid[3] - The cell centroid
$  PetscReal   volume      - The cell volume

.seealso: `DMPlexComputeGeometryFVM()`
S*/
typedef struct {
  PetscReal centroid[3];
  PetscReal volume;
} PetscFVCellGeom;

#endif
