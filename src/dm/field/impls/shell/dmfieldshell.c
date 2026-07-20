#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/

typedef struct _n_DMField_Shell {
  PetscCtx ctx;
  PetscErrorCode (*destroy)(DMField);
} DMField_Shell;

/*@
  DMFieldShellGetContext - Retrieve the user-supplied context associated with a `DMFIELDSHELL`.

  Not Collective

  Input Parameter:
. field - the `DMField` of type `DMFIELDSHELL`

  Output Parameter:
. ctx - the context pointer that was passed to `DMFieldCreateShell()`

  Level: intermediate

.seealso: `DMField`, `DMFIELDSHELL`, `DMFieldCreateShell()`
@*/
PetscErrorCode DMFieldShellGetContext(DMField field, PetscCtxRt ctx)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  PetscAssertPointer(ctx, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)field, DMFIELDSHELL, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)field), PETSC_ERR_SUP, "Cannot get context from non-shell shield");
  *(void **)ctx = ((DMField_Shell *)field->data)->ctx;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldDestroy_Shell(DMField field)
{
  DMField_Shell *shell = (DMField_Shell *)field->data;

  PetscFunctionBegin;
  if (shell->destroy) PetscCall((*shell->destroy)(field));
  PetscCall(PetscFree(field->data));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMFieldShellEvaluateFEDefault - Default finite-element evaluation for a `DMFIELDSHELL` that maps the quadrature points to real space using the coordinate `DMField` and then calls `DMFieldEvaluate()`.

  Not Collective

  Input Parameters:
+ field   - the `DMField` of type `DMFIELDSHELL`
. pointIS - the `IS` of mesh points at which to evaluate
. quad    - the reference-element quadrature
- type    - `PETSC_SCALAR` or `PETSC_REAL`

  Output Parameters:
+ B - values at quadrature points, or `NULL`
. D - derivatives at quadrature points, or `NULL`
- H - Hessians at quadrature points, or `NULL`

  Level: developer

  Note:
  Intended to be registered as the FE evaluation callback via `DMFieldShellSetEvaluateFE()` when the shell only supplies a bulk `DMFieldEvaluate()` implementation.

.seealso: `DMField`, `DMFIELDSHELL`, `DMFieldShellSetEvaluateFE()`, `DMFieldShellEvaluateFVDefault()`, `DMFieldEvaluate()`
@*/
PetscErrorCode DMFieldShellEvaluateFEDefault(DMField field, IS pointIS, PetscQuadrature quad, PetscDataType type, void *B, void *D, void *H)
{
  DM           dm = field->dm;
  DMField      coordField;
  PetscFEGeom *geom;
  Vec          pushforward;
  PetscInt     dimC, dim, numPoints, Nq, p, Nc;
  PetscScalar *pfArray;

  PetscFunctionBegin;
  Nc = field->numComponents;
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMFieldCreateFEGeom(coordField, pointIS, quad, PETSC_FEGEOM_BASIC, &geom));
  PetscCall(DMGetCoordinateDim(dm, &dimC));
  PetscCall(PetscQuadratureGetData(quad, &dim, NULL, &Nq, NULL, NULL));
  PetscCall(ISGetLocalSize(pointIS, &numPoints));
  PetscCall(PetscMalloc1(dimC * Nq * numPoints, &pfArray));
  for (p = 0; p < numPoints * dimC * Nq; p++) pfArray[p] = geom->v[p];
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)pointIS), dimC, dimC * Nq * numPoints, PETSC_DETERMINE, pfArray, &pushforward));
  PetscCall(DMFieldEvaluate(field, pushforward, type, B, D, H));
  /* TODO: handle covariant/contravariant pullbacks */
  if (D) {
    if (type == PETSC_SCALAR) {
      PetscScalar *sD = (PetscScalar *)D;
      PetscInt     q;

      for (p = 0; p < numPoints * Nq; p++) {
        for (q = 0; q < Nc; q++) {
          PetscScalar d[3];

          PetscInt i, j;

          for (i = 0; i < dimC; i++) d[i] = sD[(p * Nc + q) * dimC + i];
          for (i = 0; i < dimC; i++) sD[(p * Nc + q) * dimC + i] = 0.;
          for (i = 0; i < dimC; i++) {
            for (j = 0; j < dimC; j++) sD[(p * Nc + q) * dimC + i] += geom->J[(p * dimC + j) * dimC + i] * d[j];
          }
        }
      }
    } else {
      PetscReal *rD = (PetscReal *)D;
      PetscInt   q;

      for (p = 0; p < numPoints * Nq; p++) {
        for (q = 0; q < Nc; q++) {
          PetscReal d[3];

          PetscInt i, j;

          for (i = 0; i < dimC; i++) d[i] = rD[(p * Nc + q) * dimC + i];
          for (i = 0; i < dimC; i++) rD[(p * Nc + q) * dimC + i] = 0.;
          for (i = 0; i < dimC; i++) {
            for (j = 0; j < dimC; j++) rD[(p * Nc + q) * dimC + i] += geom->J[(p * dimC + j) * dimC + i] * d[j];
          }
        }
      }
    }
  }
  if (H) {
    if (type == PETSC_SCALAR) {
      PetscScalar *sH = (PetscScalar *)H;
      PetscInt     q;

      for (p = 0; p < numPoints * Nq; p++) {
        for (q = 0; q < Nc; q++) {
          PetscScalar d[3][3];

          PetscInt i, j, k, l;

          for (i = 0; i < dimC; i++)
            for (j = 0; j < dimC; j++) d[i][j] = sH[((p * Nc + q) * dimC + i) * dimC + j];
          for (i = 0; i < dimC; i++)
            for (j = 0; j < dimC; j++) sH[((p * Nc + q) * dimC + i) * dimC + j] = 0.;
          for (i = 0; i < dimC; i++) {
            for (j = 0; j < dimC; j++) {
              for (k = 0; k < dimC; k++) {
                for (l = 0; l < dimC; l++) sH[((p * Nc + q) * dimC + i) * dimC + j] += geom->J[(p * dimC + k) * dimC + i] * geom->J[(p * dimC + l) * dimC + j] * d[k][l];
              }
            }
          }
        }
      }
    } else {
      PetscReal *rH = (PetscReal *)H;
      PetscInt   q;

      for (p = 0; p < numPoints * Nq; p++) {
        for (q = 0; q < Nc; q++) {
          PetscReal d[3][3];

          PetscInt i, j, k, l;

          for (i = 0; i < dimC; i++)
            for (j = 0; j < dimC; j++) d[i][j] = rH[((p * Nc + q) * dimC + i) * dimC + j];
          for (i = 0; i < dimC; i++)
            for (j = 0; j < dimC; j++) rH[((p * Nc + q) * dimC + i) * dimC + j] = 0.;
          for (i = 0; i < dimC; i++) {
            for (j = 0; j < dimC; j++) {
              for (k = 0; k < dimC; k++) {
                for (l = 0; l < dimC; l++) rH[((p * Nc + q) * dimC + i) * dimC + j] += geom->J[(p * dimC + k) * dimC + i] * geom->J[(p * dimC + l) * dimC + j] * d[k][l];
              }
            }
          }
        }
      }
    }
  }
  PetscCall(VecDestroy(&pushforward));
  PetscCall(PetscFree(pfArray));
  PetscCall(PetscFEGeomDestroy(&geom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMFieldShellEvaluateFVDefault - Default finite-volume evaluation for a `DMFIELDSHELL` that samples at cell centroids using the coordinate `DMField`'s default quadrature and calls `DMFieldEvaluate()`.

  Not Collective

  Input Parameters:
+ field   - the `DMField` of type `DMFIELDSHELL`
. pointIS - the `IS` of mesh cells at which to evaluate
- type    - `PETSC_SCALAR` or `PETSC_REAL`

  Output Parameters:
+ B - cell-averaged values, or `NULL`
. D - cell-averaged derivatives, or `NULL`
- H - cell-averaged Hessians, or `NULL`

  Level: developer

  Note:
  Intended to be registered as the FV evaluation callback via `DMFieldShellSetEvaluateFV()` when the shell only supplies a bulk `DMFieldEvaluate()` implementation.

.seealso: `DMField`, `DMFIELDSHELL`, `DMFieldShellSetEvaluateFV()`, `DMFieldShellEvaluateFEDefault()`, `DMFieldEvaluate()`
@*/
PetscErrorCode DMFieldShellEvaluateFVDefault(DMField field, IS pointIS, PetscDataType type, void *B, void *D, void *H)
{
  DM              dm = field->dm;
  DMField         coordField;
  PetscFEGeom    *geom;
  Vec             pushforward;
  PetscInt        dimC, dim, numPoints, Nq, p;
  PetscScalar    *pfArray;
  PetscQuadrature quad;
  MPI_Comm        comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)field, &comm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &dimC));
  PetscCall(DMGetCoordinateField(dm, &coordField));
  PetscCall(DMFieldGetFVQuadrature_Internal(coordField, pointIS, &quad));
  PetscCheck(quad, comm, PETSC_ERR_ARG_WRONGSTATE, "coordinate field must have default quadrature for FV computation");
  PetscCall(PetscQuadratureGetData(quad, NULL, NULL, &Nq, NULL, NULL));
  PetscCheck(Nq == 1, comm, PETSC_ERR_ARG_WRONGSTATE, "quadrature must have only one point");
  PetscCall(DMFieldCreateFEGeom(coordField, pointIS, quad, PETSC_FEGEOM_BASIC, &geom));
  PetscCall(ISGetLocalSize(pointIS, &numPoints));
  PetscCall(PetscMalloc1(dimC * numPoints, &pfArray));
  for (p = 0; p < numPoints * dimC; p++) pfArray[p] = geom->v[p];
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)pointIS), dimC, dimC * numPoints, PETSC_DETERMINE, pfArray, &pushforward));
  PetscCall(DMFieldEvaluate(field, pushforward, type, B, D, H));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(VecDestroy(&pushforward));
  PetscCall(PetscFree(pfArray));
  PetscCall(PetscFEGeomDestroy(&geom));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMFieldShellSetDestroy - Register a destroy callback that will be invoked when a `DMFIELDSHELL` is destroyed.

  Logically Collective

  Input Parameters:
+ field   - the `DMField` of type `DMFIELDSHELL`
- destroy - the destroy routine, called before the shell's own data is freed

  Calling sequence of `destroy`:
. field - the `DMField` of type `DMFIELDSHELL` being destroyed

  Level: intermediate

.seealso: `DMField`, `DMFIELDSHELL`, `DMFieldCreateShell()`, `DMFieldDestroy()`
@*/
PetscErrorCode DMFieldShellSetDestroy(DMField field, PetscErrorCode (*destroy)(DMField field))
{
  DMField_Shell *shell = (DMField_Shell *)field->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  shell->destroy = destroy;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMFieldShellSetEvaluate - Register the routine that evaluates a `DMFIELDSHELL` at an arbitrary set of real-space points supplied as a `Vec` of coordinates.

  Logically Collective

  Input Parameters:
+ field    - the `DMField` of type `DMFIELDSHELL`
- evaluate - the evaluation callback

  Calling sequence of `evaluate`:
+ field - the `DMField` of type `DMFIELDSHELL`
. u     - the points at which to evaluate the field, as a `Vec` of coordinates of size d x n
. dtype - `PETSC_SCALAR` or `PETSC_REAL`
. B     - array of field values at each point, or `NULL`
. D     - array of field spatial derivatives at each point, or `NULL`
- H     - array of field spatial Hessians at each point, or `NULL`

  Level: intermediate

.seealso: `DMField`, `DMFIELDSHELL`, `DMFieldCreateShell()`, `DMFieldEvaluate()`, `DMFieldShellSetEvaluateFE()`, `DMFieldShellSetEvaluateFV()`
@*/
PetscErrorCode DMFieldShellSetEvaluate(DMField field, PetscErrorCode (*evaluate)(DMField field, Vec u, PetscDataType dtype, void *B, void *D, void *H))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  field->ops->evaluate = evaluate;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMFieldShellSetEvaluateFE - Register the routine that evaluates a `DMFIELDSHELL` at finite-element quadrature points over a set of mesh points.

  Logically Collective

  Input Parameters:
+ field      - the `DMField` of type `DMFIELDSHELL`
- evaluateFE - the FE evaluation callback

  Calling sequence of `evaluateFE`:
+ field - the `DMField` of type `DMFIELDSHELL`
. is    - the `IS` of mesh cells on which to evaluate the field
. quad  - the reference-cell `PetscQuadrature` supplying the evaluation points
. dtype - `PETSC_SCALAR` or `PETSC_REAL`
. B     - array of field values at each quadrature point, or `NULL`
. D     - array of field reference derivatives at each quadrature point, or `NULL`
- H     - array of field reference Hessians at each quadrature point, or `NULL`

  Level: intermediate

  Note:
  If the shell only supplies a generic `DMFieldEvaluate()` via `DMFieldShellSetEvaluate()`, pass `DMFieldShellEvaluateFEDefault()` here.

.seealso: `DMField`, `DMFIELDSHELL`, `DMFieldCreateShell()`, `DMFieldEvaluateFE()`, `DMFieldShellEvaluateFEDefault()`, `DMFieldShellSetEvaluateFV()`
@*/
PetscErrorCode DMFieldShellSetEvaluateFE(DMField field, PetscErrorCode (*evaluateFE)(DMField field, IS is, PetscQuadrature quad, PetscDataType dtype, void *B, void *D, void *H))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  field->ops->evaluateFE = evaluateFE;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMFieldShellSetEvaluateFV - Register the routine that evaluates a `DMFIELDSHELL` as cell averages over a set of mesh cells.

  Logically Collective

  Input Parameters:
+ field      - the `DMField` of type `DMFIELDSHELL`
- evaluateFV - the FV evaluation callback

  Calling sequence of `evaluateFV`:
+ field - the `DMField` of type `DMFIELDSHELL`
. is    - the `IS` of mesh cells on which to evaluate the field
. dtype - `PETSC_SCALAR` or `PETSC_REAL`
. B     - array of cell-averaged field values, or `NULL`
. D     - array of cell-averaged field derivatives, or `NULL`
- H     - array of cell-averaged field Hessians, or `NULL`

  Level: intermediate

  Note:
  If the shell only supplies a generic `DMFieldEvaluate()` via `DMFieldShellSetEvaluate()`, pass `DMFieldShellEvaluateFVDefault()` here.

.seealso: `DMField`, `DMFIELDSHELL`, `DMFieldCreateShell()`, `DMFieldEvaluateFV()`, `DMFieldShellEvaluateFVDefault()`, `DMFieldShellSetEvaluateFE()`
@*/
PetscErrorCode DMFieldShellSetEvaluateFV(DMField field, PetscErrorCode (*evaluateFV)(DMField field, IS is, PetscDataType dtype, void *B, void *D, void *H))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  field->ops->evaluateFV = evaluateFV;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMFieldShellSetGetDegree - Register the routine that reports the polynomial degree bounds of a `DMFIELDSHELL` over a set of mesh points.

  Logically Collective

  Input Parameters:
+ field     - the `DMField` of type `DMFIELDSHELL`
- getDegree - callback that returns the minimum and maximum polynomial degrees of the field over the given point `IS`

  Calling sequence of `getDegree`:
+ field     - the `DMField` of type `DMFIELDSHELL`
. is        - the `IS` of mesh points over which the degree bounds are requested
. minDegree - the degree of the largest polynomial space contained in the field on each element
- maxDegree - the largest degree of the smallest polynomial space containing the field on any element

  Level: intermediate

.seealso: `DMField`, `DMFIELDSHELL`, `DMFieldCreateShell()`, `DMFieldGetDegree()`
@*/
PetscErrorCode DMFieldShellSetGetDegree(DMField field, PetscErrorCode (*getDegree)(DMField field, IS is, PetscInt *minDegree, PetscInt *maxDegree))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  field->ops->getDegree = getDegree;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMFieldShellSetCreateDefaultQuadrature - Register the routine that supplies a default `PetscQuadrature` sufficient to integrate a `DMFIELDSHELL` exactly over a set of mesh points.

  Logically Collective

  Input Parameters:
+ field  - the `DMField` of type `DMFIELDSHELL`
- create - callback that returns a newly created `PetscQuadrature` for the given point `IS`

  Calling sequence of `create`:
+ f    - the `DMField` of type `DMFIELDSHELL`
. is   - the `IS` of mesh points over which the field will be integrated
- quad - the newly created `PetscQuadrature`

  Level: intermediate

.seealso: `DMField`, `DMFIELDSHELL`, `DMFieldCreateShell()`, `DMFieldCreateDefaultQuadrature()`
@*/
PetscErrorCode DMFieldShellSetCreateDefaultQuadrature(DMField field, PetscErrorCode (*create)(DMField f, IS is, PetscQuadrature *quad))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  field->ops->createDefaultQuadrature = create;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMFieldInitialize_Shell(DMField field)
{
  PetscFunctionBegin;
  field->ops->destroy                 = DMFieldDestroy_Shell;
  field->ops->evaluate                = NULL;
  field->ops->evaluateFE              = DMFieldShellEvaluateFEDefault;
  field->ops->evaluateFV              = DMFieldShellEvaluateFVDefault;
  field->ops->getDegree               = NULL;
  field->ops->createDefaultQuadrature = NULL;
  field->ops->view                    = NULL;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_INTERN PetscErrorCode DMFieldCreate_Shell(DMField field)
{
  DMField_Shell *shell;

  PetscFunctionBegin;
  PetscCall(PetscNew(&shell));
  field->data = shell;
  PetscCall(DMFieldInitialize_Shell(field));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMFieldCreateShell - Create a `DMFIELDSHELL`, a `DMField` whose evaluation is implemented entirely by user-supplied callbacks.

  Collective

  Input Parameters:
+ dm            - the `DM` on which the field lives
. numComponents - the number of components of the field
. continuity    - the continuity of the field (e.g. `DMFIELD_VERTEX`)
- ctx           - optional application context returned by `DMFieldShellGetContext()`

  Output Parameter:
. field - the newly created `DMField` of type `DMFIELDSHELL`

  Level: intermediate

  Note:
  After creation the user must register the desired evaluation callbacks with `DMFieldShellSetEvaluate()`, `DMFieldShellSetEvaluateFE()`, `DMFieldShellSetEvaluateFV()`, and optionally `DMFieldShellSetDestroy()`, `DMFieldShellSetGetDegree()`, and `DMFieldShellSetCreateDefaultQuadrature()`.

.seealso: `DMField`, `DMFIELDSHELL`, `DMFieldShellGetContext()`, `DMFieldShellSetEvaluate()`, `DMFieldShellSetEvaluateFE()`, `DMFieldShellSetEvaluateFV()`, `DMFieldShellSetDestroy()`
@*/
PetscErrorCode DMFieldCreateShell(DM dm, PetscInt numComponents, DMFieldContinuity continuity, PetscCtx ctx, DMField *field)
{
  DMField        b;
  DMField_Shell *shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (ctx) PetscAssertPointer(ctx, 4);
  PetscAssertPointer(field, 5);
  PetscCall(DMFieldCreate(dm, numComponents, continuity, &b));
  PetscCall(DMFieldSetType(b, DMFIELDSHELL));
  shell      = (DMField_Shell *)b->data;
  shell->ctx = ctx;
  *field     = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}
