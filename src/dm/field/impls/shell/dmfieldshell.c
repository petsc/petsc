#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/

typedef struct _n_DMField_Shell {
  void *ctx;
  PetscErrorCode (*destroy)(DMField);
} DMField_Shell;

PetscErrorCode DMFieldShellGetContext(DMField field, void *ctx)
{
  PetscBool flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  PetscValidPointer(ctx, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)field, DMFIELDSHELL, &flg));
  if (flg) *(void **)ctx = ((DMField_Shell *)(field->data))->ctx;
  else SETERRQ(PetscObjectComm((PetscObject)field), PETSC_ERR_SUP, "Cannot get context from non-shell shield");
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldDestroy_Shell(DMField field)
{
  DMField_Shell *shell = (DMField_Shell *)field->data;

  PetscFunctionBegin;
  if (shell->destroy) PetscCall((*(shell->destroy))(field));
  PetscCall(PetscFree(field->data));
  PetscFunctionReturn(0);
}

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
  PetscCall(DMFieldCreateFEGeom(coordField, pointIS, quad, PETSC_FALSE, &geom));
  PetscCall(DMGetCoordinateDim(dm, &dimC));
  PetscCall(PetscQuadratureGetData(quad, &dim, NULL, &Nq, NULL, NULL));
  PetscCall(ISGetLocalSize(pointIS, &numPoints));
  PetscCall(PetscMalloc1(dimC * Nq * numPoints, &pfArray));
  for (p = 0; p < numPoints * dimC * Nq; p++) pfArray[p] = (PetscScalar)geom->v[p];
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
  PetscFunctionReturn(0);
}

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
  PetscCall(DMFieldCreateFEGeom(coordField, pointIS, quad, PETSC_FALSE, &geom));
  PetscCall(ISGetLocalSize(pointIS, &numPoints));
  PetscCall(PetscMalloc1(dimC * numPoints, &pfArray));
  for (p = 0; p < numPoints * dimC; p++) pfArray[p] = (PetscScalar)geom->v[p];
  PetscCall(VecCreateMPIWithArray(PetscObjectComm((PetscObject)pointIS), dimC, dimC * numPoints, PETSC_DETERMINE, pfArray, &pushforward));
  PetscCall(DMFieldEvaluate(field, pushforward, type, B, D, H));
  PetscCall(PetscQuadratureDestroy(&quad));
  PetscCall(VecDestroy(&pushforward));
  PetscCall(PetscFree(pfArray));
  PetscCall(PetscFEGeomDestroy(&geom));
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetDestroy(DMField field, PetscErrorCode (*destroy)(DMField))
{
  DMField_Shell *shell = (DMField_Shell *)field->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  shell->destroy = destroy;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetEvaluate(DMField field, PetscErrorCode (*evaluate)(DMField, Vec, PetscDataType, void *, void *, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  field->ops->evaluate = evaluate;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetEvaluateFE(DMField field, PetscErrorCode (*evaluateFE)(DMField, IS, PetscQuadrature, PetscDataType, void *, void *, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  field->ops->evaluateFE = evaluateFE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetEvaluateFV(DMField field, PetscErrorCode (*evaluateFV)(DMField, IS, PetscDataType, void *, void *, void *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  field->ops->evaluateFV = evaluateFV;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetGetDegree(DMField field, PetscErrorCode (*getDegree)(DMField, IS, PetscInt *, PetscInt *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  field->ops->getDegree = getDegree;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetCreateDefaultQuadrature(DMField field, PetscErrorCode (*createDefaultQuadrature)(DMField, IS, PetscQuadrature *))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID, 1);
  field->ops->createDefaultQuadrature = createDefaultQuadrature;
  PetscFunctionReturn(0);
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
  PetscFunctionReturn(0);
}

PETSC_INTERN PetscErrorCode DMFieldCreate_Shell(DMField field)
{
  DMField_Shell *shell;

  PetscFunctionBegin;
  PetscCall(PetscNew(&shell));
  field->data = shell;
  PetscCall(DMFieldInitialize_Shell(field));
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldCreateShell(DM dm, PetscInt numComponents, DMFieldContinuity continuity, void *ctx, DMField *field)
{
  DMField        b;
  DMField_Shell *shell;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm, DM_CLASSID, 1);
  if (ctx) PetscValidPointer(ctx, 4);
  PetscValidPointer(field, 5);
  PetscCall(DMFieldCreate(dm, numComponents, continuity, &b));
  PetscCall(DMFieldSetType(b, DMFIELDSHELL));
  shell      = (DMField_Shell *)b->data;
  shell->ctx = ctx;
  *field     = b;
  PetscFunctionReturn(0);
}
