#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/

typedef struct _n_DMField_Shell
{
  void *ctx;
  PetscErrorCode (*destroy) (DMField);
}
DMField_Shell;

PetscErrorCode DMFieldShellGetContext(DMField field, void *ctx)
{
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidPointer(ctx,2);
  ierr = PetscObjectTypeCompare((PetscObject)field,DMFIELDSHELL,&flg);CHKERRQ(ierr);
  if (flg) *(void**)ctx = ((DMField_Shell*)(field->data))->ctx;
  else SETERRQ(PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Cannot get context from non-shell shield");
  PetscFunctionReturn(0);
}

static PetscErrorCode DMFieldDestroy_Shell(DMField field)
{
  DMField_Shell *shell = (DMField_Shell *) field->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (shell->destroy) {ierr = (*(shell->destroy)) (field);CHKERRQ(ierr);}
  ierr = PetscFree(field->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellEvaluateFEDefault(DMField field, IS pointIS, PetscQuadrature quad, PetscDataType type, void *B, void *D, void *H)
{
  DM              dm = field->dm;
  DMField         coordField;
  PetscFEGeom    *geom;
  Vec             pushforward;
  PetscInt        dimC, dim, numPoints, Nq, p, Nc;
  PetscScalar    *pfArray;
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  Nc   = field->numComponents;
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMFieldCreateFEGeom(coordField, pointIS, quad, PETSC_FALSE, &geom);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimC);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, &dim, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
  ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
  ierr = PetscMalloc1(dimC * Nq * numPoints, &pfArray);CHKERRQ(ierr);
  for (p = 0; p < numPoints * dimC * Nq; p++) pfArray[p] = (PetscScalar) geom->v[p];
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pointIS), dimC, dimC * Nq * numPoints, PETSC_DETERMINE, pfArray, &pushforward);CHKERRQ(ierr);
  ierr = DMFieldEvaluate(field, pushforward, type, B, D, H);CHKERRQ(ierr);
  /* TODO: handle covariant/contravariant pullbacks */
  if (D) {
    if (type == PETSC_SCALAR) {
      PetscScalar *sD = (PetscScalar *) D;
      PetscInt q;

      for (p = 0; p < numPoints * Nq; p++) {
        for (q = 0; q < Nc; q++) {
          PetscScalar d[3];

          PetscInt i, j;

          for (i = 0; i < dimC; i++) d[i] = sD[(p * Nc + q) * dimC + i];
          for (i = 0; i < dimC; i++) sD[(p * Nc + q) * dimC + i] = 0.;
          for (i = 0; i < dimC; i++) {
            for (j = 0; j < dimC; j++) {
              sD[(p * Nc + q) * dimC + i] += geom->J[(p * dimC + j) * dimC + i] * d[j];
            }
          }
        }
      }
    } else {
      PetscReal *rD = (PetscReal *) D;
      PetscInt q;

      for (p = 0; p < numPoints * Nq; p++) {
        for (q = 0; q < Nc; q++) {
          PetscReal d[3];

          PetscInt i, j;

          for (i = 0; i < dimC; i++) d[i] = rD[(p * Nc + q) * dimC + i];
          for (i = 0; i < dimC; i++) rD[(p * Nc + q) * dimC + i] = 0.;
          for (i = 0; i < dimC; i++) {
            for (j = 0; j < dimC; j++) {
              rD[(p * Nc + q) * dimC + i] += geom->J[(p * dimC + j) * dimC + i] * d[j];
            }
          }
        }
      }
    }
  }
  if (H) {
    if (type == PETSC_SCALAR) {
      PetscScalar *sH = (PetscScalar *) H;
      PetscInt q;

      for (p = 0; p < numPoints * Nq; p++) {
        for (q = 0; q < Nc; q++) {
          PetscScalar d[3][3];

          PetscInt i, j, k, l;

          for (i = 0; i < dimC; i++) for (j = 0; j < dimC; j++) d[i][j] = sH[((p * Nc + q) * dimC + i) * dimC + j];
          for (i = 0; i < dimC; i++) for (j = 0; j < dimC; j++) sH[((p * Nc + q) * dimC + i) * dimC + j] = 0.;
          for (i = 0; i < dimC; i++) {
            for (j = 0; j < dimC; j++) {
              for (k = 0; k < dimC; k++) {
                for (l = 0; l < dimC; l++) {
                  sH[((p * Nc + q) * dimC + i) * dimC + j] += geom->J[(p * dimC + k) * dimC + i] * geom->J[(p * dimC + l) * dimC + j] * d[k][l];
                }
              }
            }
          }
        }
      }
    } else {
      PetscReal *rH = (PetscReal *) H;
      PetscInt q;

      for (p = 0; p < numPoints * Nq; p++) {
        for (q = 0; q < Nc; q++) {
          PetscReal d[3][3];

          PetscInt i, j, k, l;

          for (i = 0; i < dimC; i++) for (j = 0; j < dimC; j++) d[i][j] = rH[((p * Nc + q) * dimC + i) * dimC + j];
          for (i = 0; i < dimC; i++) for (j = 0; j < dimC; j++) rH[((p * Nc + q) * dimC + i) * dimC + j] = 0.;
          for (i = 0; i < dimC; i++) {
            for (j = 0; j < dimC; j++) {
              for (k = 0; k < dimC; k++) {
                for (l = 0; l < dimC; l++) {
                  rH[((p * Nc + q) * dimC + i) * dimC + j] += geom->J[(p * dimC + k) * dimC + i] * geom->J[(p * dimC + l) * dimC + j] * d[k][l];
                }
              }
            }
          }
        }
      }
    }
  }
  ierr = VecDestroy(&pushforward);CHKERRQ(ierr);
  ierr = PetscFree(pfArray);CHKERRQ(ierr);
  ierr = PetscFEGeomDestroy(&geom);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  ierr = DMGetCoordinateField(dm, &coordField);CHKERRQ(ierr);
  ierr = DMFieldCreateDefaultQuadrature(coordField, pointIS, &quad);CHKERRQ(ierr);
  PetscCheckFalse(!quad,PetscObjectComm((PetscObject) pointIS), PETSC_ERR_ARG_WRONGSTATE, "coordinate field must have default quadrature for FV computation");
  ierr = DMFieldCreateFEGeom(coordField, pointIS, quad, PETSC_FALSE, &geom);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &dimC);CHKERRQ(ierr);
  ierr = PetscQuadratureGetData(quad, &dim, NULL, &Nq, NULL, NULL);CHKERRQ(ierr);
  PetscCheckFalse(Nq != 1,PetscObjectComm((PetscObject) quad), PETSC_ERR_ARG_WRONGSTATE, "quadrature must have only one point");
  ierr = ISGetLocalSize(pointIS, &numPoints);CHKERRQ(ierr);
  ierr = PetscMalloc1(dimC * numPoints, &pfArray);CHKERRQ(ierr);
  for (p = 0; p < numPoints * dimC; p++) pfArray[p] = (PetscScalar) geom->v[p];
  ierr = VecCreateMPIWithArray(PetscObjectComm((PetscObject)pointIS), dimC, dimC * numPoints, PETSC_DETERMINE, pfArray, &pushforward);CHKERRQ(ierr);
  ierr = DMFieldEvaluate(field, pushforward, type, B, D, H);CHKERRQ(ierr);
  ierr = PetscQuadratureDestroy(&quad);CHKERRQ(ierr);
  ierr = VecDestroy(&pushforward);CHKERRQ(ierr);
  ierr = PetscFree(pfArray);CHKERRQ(ierr);
  ierr = PetscFEGeomDestroy(&geom);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetDestroy(DMField field, PetscErrorCode (*destroy)(DMField))
{
  DMField_Shell *shell = (DMField_Shell *) field->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  shell->destroy = destroy;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetEvaluate(DMField field, PetscErrorCode (*evaluate)(DMField,Vec,PetscDataType,void*,void*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  field->ops->evaluate = evaluate;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetEvaluateFE(DMField field, PetscErrorCode (*evaluateFE)(DMField,IS,PetscQuadrature,PetscDataType,void*,void*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  field->ops->evaluateFE = evaluateFE;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetEvaluateFV(DMField field, PetscErrorCode (*evaluateFV)(DMField,IS,PetscDataType,void*,void*,void*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  field->ops->evaluateFV = evaluateFV;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetGetDegree(DMField field, PetscErrorCode (*getDegree)(DMField,IS,PetscInt*,PetscInt*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  field->ops->getDegree = getDegree;
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldShellSetCreateDefaultQuadrature(DMField field, PetscErrorCode (*createDefaultQuadrature)(DMField,IS,PetscQuadrature*))
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(field,&shell);CHKERRQ(ierr);
  field->data = shell;
  ierr = DMFieldInitialize_Shell(field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode DMFieldCreateShell(DM dm, PetscInt numComponents, DMFieldContinuity continuity, void *ctx, DMField *field)
{
  DMField        b;
  DMField_Shell  *shell;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  if (ctx) PetscValidPointer(ctx, 4);
  PetscValidPointer(field, 5);
  ierr = DMFieldCreate(dm,numComponents,continuity,&b);CHKERRQ(ierr);
  ierr = DMFieldSetType(b,DMFIELDSHELL);CHKERRQ(ierr);
  shell = (DMField_Shell *) b->data;
  shell->ctx = ctx;
  *field = b;
  PetscFunctionReturn(0);
}
