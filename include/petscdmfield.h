#pragma once

#include <petscdm.h>
#include <petscdt.h>
#include <petscfe.h>

/* SUBMANSEC = DM */

PETSC_EXTERN PetscErrorCode DMFieldInitializePackage(void);
PETSC_EXTERN PetscErrorCode DMFieldFinalizePackage(void);

PETSC_EXTERN PetscClassId DMFIELD_CLASSID;

/*J
  DMFieldType - String with the name of a `DMField` implementation

  Values:
+ `DMFIELDDA`    - a field defined only by its values at the corners of a `DMDA`
. `DMFIELDDS`    - a field defined by a discretization over a mesh set with `DMSetField()`
- `DMFIELDSHELL` - a field defined by arbitrary callbacks

  Level: intermediate

.seealso: [](ch_dmbase), `DMField`, `DMFieldSetType()`, `DMFieldGetType()`, `DMFieldRegister()`
J*/
typedef const char *DMFieldType;
#define DMFIELDDA    "da"
#define DMFIELDDS    "ds"
#define DMFIELDSHELL "shell"

PETSC_EXTERN PetscFunctionList DMFieldList;
PETSC_EXTERN PetscErrorCode    DMFieldSetType(DMField, DMFieldType);
PETSC_EXTERN PetscErrorCode    DMFieldGetType(DMField, DMFieldType *);
PETSC_EXTERN PetscErrorCode    DMFieldRegister(const char[], PetscErrorCode (*)(DMField));

/*E
   DMFieldContinuity - Indicates the smallest mesh entity across which a `DMField` is continuous; equivalently, the largest entity at which the field may be discontinuous

   Values:
+   `DMFIELD_VERTEX` - continuous across vertices (i.e., everywhere on the mesh; standard $H^1$ finite elements)
.   `DMFIELD_EDGE`   - continuous across edges, but may jump at vertices
.   `DMFIELD_FACET`  - continuous across facets (faces in 3D, edges in 2D); may jump at lower-dimensional points
-   `DMFIELD_CELL`   - field is defined per cell, with no continuity between adjacent cells (cell-centered finite volume)

   Level: intermediate

.seealso: `DMField`, `DMFieldCreateShell()`, `DMFieldEvaluate()`, `DMFieldEvaluateFE()`, `DMFieldEvaluateFV()`
E*/
typedef enum {
  DMFIELD_VERTEX,
  DMFIELD_EDGE,
  DMFIELD_FACET,
  DMFIELD_CELL
} DMFieldContinuity;
PETSC_EXTERN const char *const DMFieldContinuities[];

PETSC_EXTERN PetscErrorCode DMFieldDestroy(DMField *);
PETSC_EXTERN PetscErrorCode DMFieldView(DMField, PetscViewer);

PETSC_EXTERN PetscErrorCode DMFieldGetDM(DMField, DM *);
PETSC_EXTERN PetscErrorCode DMFieldGetNumComponents(DMField, PetscInt *);

PETSC_EXTERN PetscErrorCode DMFieldEvaluate(DMField, Vec, PetscDataType, void *, void *, void *);
PETSC_EXTERN PetscErrorCode DMFieldEvaluateFE(DMField, IS, PetscQuadrature, PetscDataType, void *, void *, void *);
PETSC_EXTERN PetscErrorCode DMFieldEvaluateFV(DMField, IS, PetscDataType, void *, void *, void *);
PETSC_EXTERN PetscErrorCode DMFieldCreateFEGeom(DMField, IS, PetscQuadrature, PetscFEGeomMode, PetscFEGeom **);

PETSC_EXTERN PetscErrorCode DMFieldCreateDefaultQuadrature(DMField, IS, PetscQuadrature *);
PETSC_EXTERN PetscErrorCode DMFieldCreateDefaultFaceQuadrature(DMField, IS, PetscQuadrature *);

PETSC_EXTERN PetscErrorCode DMFieldGetDegree(DMField, IS, PetscInt *, PetscInt *);

PETSC_EXTERN PetscErrorCode DMFieldCreateDA(DM, PetscInt, const PetscScalar *, DMField *);
PETSC_EXTERN PetscErrorCode DMFieldCreateDS(DM, PetscInt, Vec, DMField *);
PETSC_EXTERN PetscErrorCode DMFieldCreateDSWithDG(DM, DM, PetscInt, Vec, Vec, DMField *);

PETSC_EXTERN PetscErrorCode DMFieldCreateShell(DM, PetscInt, DMFieldContinuity, PetscCtx, DMField *);
PETSC_EXTERN PetscErrorCode DMFieldShellSetDestroy(DMField, PetscErrorCode (*)(DMField));
PETSC_EXTERN PetscErrorCode DMFieldShellGetContext(DMField, PetscCtxRt);
PETSC_EXTERN PetscErrorCode DMFieldShellSetEvaluate(DMField, PetscErrorCode (*)(DMField, Vec, PetscDataType, void *, void *, void *));
PETSC_EXTERN PetscErrorCode DMFieldShellSetEvaluateFE(DMField, PetscErrorCode (*)(DMField, IS, PetscQuadrature, PetscDataType, void *, void *, void *));
PETSC_EXTERN PetscErrorCode DMFieldShellEvaluateFEDefault(DMField, IS, PetscQuadrature, PetscDataType, void *, void *, void *);
PETSC_EXTERN PetscErrorCode DMFieldShellSetEvaluateFV(DMField, PetscErrorCode (*)(DMField, IS, PetscDataType, void *, void *, void *));
PETSC_EXTERN PetscErrorCode DMFieldShellEvaluateFVDefault(DMField, IS, PetscDataType, void *, void *, void *);
PETSC_EXTERN PetscErrorCode DMFieldShellSetGetDegree(DMField, PetscErrorCode (*)(DMField, IS, PetscInt *, PetscInt *));
PETSC_EXTERN PetscErrorCode DMFieldShellSetCreateDefaultQuadrature(DMField, PetscErrorCode (*)(DMField, IS, PetscQuadrature *));
