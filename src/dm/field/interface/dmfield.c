#include <petsc/private/dmfieldimpl.h> /*I "petscdmfield.h" I*/
#include <petsc/private/petscfeimpl.h> /*I "petscdmfield.h" I*/
#include <petscdmplex.h>

const char *const DMFieldContinuities[] = {
  "VERTEX",
  "EDGE",
  "FACET",
  "CELL",
  NULL
};

PETSC_INTERN PetscErrorCode DMFieldCreate(DM dm,PetscInt numComponents,DMFieldContinuity continuity,DMField *field)
{
  PetscErrorCode ierr;
  DMField        b;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(dm,DM_CLASSID,1);
  PetscValidPointer(field,2);
  ierr = DMFieldInitializePackage();CHKERRQ(ierr);

  ierr = PetscHeaderCreate(b,DMFIELD_CLASSID,"DMField","Field over DM","DM",PetscObjectComm((PetscObject)dm),DMFieldDestroy,DMFieldView);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject)dm);CHKERRQ(ierr);
  b->dm = dm;
  b->continuity = continuity;
  b->numComponents = numComponents;
  *field = b;
  PetscFunctionReturn(0);
}

/*@
   DMFieldDestroy - destroy a DMField

   Collective

   Input Arguments:
.  field - address of DMField

   Level: advanced

.seealso: DMFieldCreate()
@*/
PetscErrorCode DMFieldDestroy(DMField *field)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!*field) PetscFunctionReturn(0);
  PetscValidHeaderSpecific((*field),DMFIELD_CLASSID,1);
  if (--((PetscObject)(*field))->refct > 0) {*field = NULL; PetscFunctionReturn(0);}
  if ((*field)->ops->destroy) {ierr = (*(*field)->ops->destroy)(*field);CHKERRQ(ierr);}
  ierr = DMDestroy(&((*field)->dm));CHKERRQ(ierr);
  ierr = PetscHeaderDestroy(field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
   DMFieldView - view a DMField

   Collective

   Input Arguments:
+  field - DMField
-  viewer - viewer to display field, for example PETSC_VIEWER_STDOUT_WORLD

   Level: advanced

.seealso: DMFieldCreate()
@*/
PetscErrorCode DMFieldView(DMField field,PetscViewer viewer)
{
  PetscErrorCode    ierr;
  PetscBool         iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  if (!viewer) {ierr = PetscViewerASCIIGetStdout(PetscObjectComm((PetscObject)field),&viewer);CHKERRQ(ierr);}
  PetscValidHeaderSpecific(viewer,PETSC_VIEWER_CLASSID,2);
  PetscCheckSameComm(field,1,viewer,2);
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscObjectPrintClassNamePrefixType((PetscObject)field,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%D components\n",field->numComponents);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"%s continuity\n",DMFieldContinuities[field->continuity]);CHKERRQ(ierr);
    ierr = PetscViewerPushFormat(viewer,PETSC_VIEWER_DEFAULT);CHKERRQ(ierr);
    ierr = DMView(field->dm,viewer);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(viewer);CHKERRQ(ierr);
  }
  if (field->ops->view) {ierr = (*field->ops->view)(field,viewer);CHKERRQ(ierr);}
  if (iascii) {
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
   DMFieldSetType - set the DMField implementation

   Collective on field

   Input Parameters:
+  field - the DMField context
-  type - a known method

   Notes:
   See "include/petscvec.h" for available methods (for instance)
+    DMFIELDDA    - a field defined only by its values at the corners of a DMDA
.    DMFIELDDS    - a field defined by a discretization over a mesh set with DMSetField()
-    DMFIELDSHELL - a field defined by arbitrary callbacks

  Level: advanced

.seealso: DMFieldType,
@*/
PetscErrorCode DMFieldSetType(DMField field,DMFieldType type)
{
  PetscErrorCode ierr,(*r)(DMField);
  PetscBool      match;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidCharPointer(type,2);

  ierr = PetscObjectTypeCompare((PetscObject)field,type,&match);CHKERRQ(ierr);
  if (match) PetscFunctionReturn(0);

  ierr = PetscFunctionListFind(DMFieldList,type,&r);CHKERRQ(ierr);
  if (!r) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_ARG_UNKNOWN_TYPE,"Unable to find requested DMField type %s",type);
  /* Destroy the previous private DMField context */
  if (field->ops->destroy) {
    ierr = (*(field)->ops->destroy)(field);CHKERRQ(ierr);
  }
  ierr = PetscMemzero(field->ops,sizeof(*field->ops));CHKERRQ(ierr);
  ierr = PetscObjectChangeTypeName((PetscObject)field,type);CHKERRQ(ierr);
  field->ops->create = r;
  ierr = (*r)(field);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  DMFieldGetType - Gets the DMField type name (as a string) from the DMField.

  Not Collective

  Input Parameter:
. field  - The DMField context

  Output Parameter:
. type - The DMField type name

  Level: advanced

.seealso: DMFieldSetType()
@*/
PetscErrorCode  DMFieldGetType(DMField field, DMFieldType *type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field, DMFIELD_CLASSID,1);
  PetscValidPointer(type,2);
  ierr = DMFieldRegisterAll();CHKERRQ(ierr);
  *type = ((PetscObject)field)->type_name;
  PetscFunctionReturn(0);
}

/*@
  DMFieldGetNumComponents - Returns the number of components in the field

  Not collective

  Input Parameter:
. field - The DMField object

  Output Parameter:
. nc - The number of field components

  Level: intermediate

.seealso: DMFieldEvaluate()
@*/
PetscErrorCode DMFieldGetNumComponents(DMField field, PetscInt *nc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidIntPointer(nc,2);
  *nc = field->numComponents;
  PetscFunctionReturn(0);
}

/*@
  DMFieldGetDM - Returns the DM for the manifold over which the field is defined.

  Not collective

  Input Parameter:
. field - The DMField object

  Output Parameter:
. dm - The DM object

  Level: intermediate

.seealso: DMFieldEvaluate()
@*/
PetscErrorCode DMFieldGetDM(DMField field, DM *dm)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidPointer(dm,2);
  *dm = field->dm;
  PetscFunctionReturn(0);
}

/*@
  DMFieldEvaluate - Evaluate the field and its derivatives on a set of points

  Collective on points

  Input Parameters:
+ field - The DMField object
. points - The points at which to evaluate the field.  Should have size d x n,
           where d is the coordinate dimension of the manifold and n is the number
           of points
- datatype - The PetscDataType of the output arrays: either PETSC_REAL or PETSC_SCALAR.
             If the field is complex and datatype is PETSC_REAL, the real part of the
             field is returned.

  Output Parameters:
+ B - pointer to data of size c * n * sizeof(datatype), where c is the number of components in the field.
      If B is not NULL, the values of the field are written in this array, varying first by component,
      then by point.
. D - pointer to data of size d * c * n * sizeof(datatype).
      If D is not NULL, the values of the field's spatial derivatives are written in this array,
      varying first by the partial derivative component, then by field component, then by point.
- H - pointer to data of size d * d * c * n * sizeof(datatype).
      If H is not NULL, the values of the field's second spatial derivatives are written in this array,
      varying first by the second partial derivative component, then by field component, then by point.

  Level: intermediate

.seealso: DMFieldGetDM(), DMFieldGetNumComponents(), DMFieldEvaluateFE(), DMFieldEvaluateFV()
@*/
PetscErrorCode DMFieldEvaluate(DMField field, Vec points, PetscDataType datatype, void *B, void *D, void *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidHeaderSpecific(points,VEC_CLASSID,2);
  if (B) PetscValidPointer(B,4);
  if (D) PetscValidPointer(D,5);
  if (H) PetscValidPointer(H,6);
  if (field->ops->evaluate) {
    ierr = (*field->ops->evaluate) (field, points, datatype, B, D, H);CHKERRQ(ierr);
  } else SETERRQ (PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented for this type");
  PetscFunctionReturn(0);
}

/*@
  DMFieldEvaluateFE - Evaluate the field and its derivatives on a set of points mapped from
  quadrature points on a reference point.  The derivatives are taken with respect to the
  reference coordinates.

  Not collective

  Input Parameters:
+ field - The DMField object
. cellIS - Index set for cells on which to evaluate the field
. points - The quadature containing the points in the reference cell at which to evaluate the field.
- datatype - The PetscDataType of the output arrays: either PETSC_REAL or PETSC_SCALAR.
             If the field is complex and datatype is PETSC_REAL, the real part of the
             field is returned.

  Output Parameters:
+ B - pointer to data of size c * n * sizeof(datatype), where c is the number of components in the field.
      If B is not NULL, the values of the field are written in this array, varying first by component,
      then by point.
. D - pointer to data of size d * c * n * sizeof(datatype).
      If D is not NULL, the values of the field's spatial derivatives are written in this array,
      varying first by the partial derivative component, then by field component, then by point.
- H - pointer to data of size d * d * c * n * sizeof(datatype).
      If H is not NULL, the values of the field's second spatial derivatives are written in this array,
      varying first by the second partial derivative component, then by field component, then by point.

  Level: intermediate

.seealso: DMFieldGetNumComponents(), DMFieldEvaluate(), DMFieldEvaluateFV()
@*/
PetscErrorCode DMFieldEvaluateFE(DMField field, IS cellIS, PetscQuadrature points, PetscDataType datatype, void *B, void *D, void *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidHeaderSpecific(cellIS,IS_CLASSID,2);
  PetscValidHeader(points,3);
  if (B) PetscValidPointer(B,5);
  if (D) PetscValidPointer(D,6);
  if (H) PetscValidPointer(H,7);
  if (field->ops->evaluateFE) {
    ierr = (*field->ops->evaluateFE) (field, cellIS, points, datatype, B, D, H);CHKERRQ(ierr);
  } else SETERRQ (PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented for this type");
  PetscFunctionReturn(0);
}

/*@
  DMFieldEvaluateFV - Evaluate the mean of a field and its finite volume derivatives on a set of points.

  Not collective

  Input Parameters:
+ field - The DMField object
. cellIS - Index set for cells on which to evaluate the field
- datatype - The PetscDataType of the output arrays: either PETSC_REAL or PETSC_SCALAR.
             If the field is complex and datatype is PETSC_REAL, the real part of the
             field is returned.

  Output Parameters:
+ B - pointer to data of size c * n * sizeof(datatype), where c is the number of components in the field.
      If B is not NULL, the values of the field are written in this array, varying first by component,
      then by point.
. D - pointer to data of size d * c * n * sizeof(datatype).
      If D is not NULL, the values of the field's spatial derivatives are written in this array,
      varying first by the partial derivative component, then by field component, then by point.
- H - pointer to data of size d * d * c * n * sizeof(datatype).
      If H is not NULL, the values of the field's second spatial derivatives are written in this array,
      varying first by the second partial derivative component, then by field component, then by point.

  Level: intermediate

.seealso: DMFieldGetNumComponents(), DMFieldEvaluate(), DMFieldEvaluateFE()
@*/
PetscErrorCode DMFieldEvaluateFV(DMField field, IS cellIS, PetscDataType datatype, void *B, void *D, void *H)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidHeaderSpecific(cellIS,IS_CLASSID,2);
  if (B) PetscValidPointer(B,4);
  if (D) PetscValidPointer(D,5);
  if (H) PetscValidPointer(H,6);
  if (field->ops->evaluateFV) {
    ierr = (*field->ops->evaluateFV) (field, cellIS, datatype, B, D, H);CHKERRQ(ierr);
  } else SETERRQ (PetscObjectComm((PetscObject)field),PETSC_ERR_SUP,"Not implemented for this type");
  PetscFunctionReturn(0);
}

/*@
  DMFieldGetDegree - Get the polynomial degree of a field when pulled back onto the
  reference element

  Not collective

  Input Arguments:
+ field - the DMField object
- cellIS - the index set of points over which we want know the invariance

  Output Arguments:
+ minDegree - the degree of the largest polynomial space contained in the field on each element
- maxDegree - the largest degree of the smallest polynomial space containing the field on any element

  Level: intermediate

.seealso: DMFieldEvaluateFE()
@*/
PetscErrorCode DMFieldGetDegree(DMField field, IS cellIS, PetscInt *minDegree, PetscInt *maxDegree)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidHeaderSpecific(cellIS,IS_CLASSID,2);
  if (minDegree) PetscValidPointer(minDegree,3);
  if (maxDegree) PetscValidPointer(maxDegree,4);

  if (minDegree) *minDegree = -1;
  if (maxDegree) *maxDegree = PETSC_MAX_INT;

  if (field->ops->getDegree) {
    ierr = (*field->ops->getDegree) (field,cellIS,minDegree,maxDegree);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@
  DMFieldCreateDefaultQuadrature - Creates a quadrature sufficient to integrate the field on the selected
  points via pullback onto the reference element

  Not collective

  Input Arguments:
+ field - the DMField object
- pointIS - the index set of points over which we wish to integrate the field

  Output Arguments:
. quad - a PetscQuadrature object

  Level: developer

.seealso: DMFieldEvaluteFE(), DMFieldGetDegree()
@*/
PetscErrorCode DMFieldCreateDefaultQuadrature(DMField field, IS pointIS, PetscQuadrature *quad)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidHeaderSpecific(pointIS,IS_CLASSID,2);
  PetscValidPointer(quad,3);

  *quad = NULL;
  if (field->ops->createDefaultQuadrature) {
    ierr = (*field->ops->createDefaultQuadrature)(field, pointIS, quad);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*@C
  DMFieldCreateFEGeom - Compute and create the geometric factors of a coordinate field

  Not collective

  Input Arguments:
+ field - the DMField object
. pointIS - the index set of points over which we wish to integrate the field
. quad - the quadrature points at which to evaluate the geometric factors
- faceData - whether additional data for facets (the normal vectors and adjacent cells) should
  be calculated

  Output Arguments:
. geom - the geometric factors

  Level: developer

.seealso: DMFieldEvaluateFE(), DMFieldCreateDefaulteQuadrature(), DMFieldGetDegree()
@*/
PetscErrorCode DMFieldCreateFEGeom(DMField field, IS pointIS, PetscQuadrature quad, PetscBool faceData, PetscFEGeom **geom)
{
  PetscInt       dim, dE;
  PetscInt       nPoints;
  PetscInt       maxDegree;
  PetscFEGeom    *g;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(field,DMFIELD_CLASSID,1);
  PetscValidHeaderSpecific(pointIS,IS_CLASSID,2);
  PetscValidHeader(quad,3);
  ierr = ISGetLocalSize(pointIS,&nPoints);CHKERRQ(ierr);
  dE = field->numComponents;
  ierr = PetscFEGeomCreate(quad,nPoints,dE,faceData,&g);CHKERRQ(ierr);
  ierr = DMFieldEvaluateFE(field,pointIS,quad,PETSC_REAL,g->v,g->J,NULL);CHKERRQ(ierr);
  dim = g->dim;
  if (dE > dim) {
    /* space out J and make square Jacobians */
    PetscInt  i, j, k, N = g->numPoints * g->numCells;

    for (i = N-1; i >= 0; i--) {
      PetscReal   J[9];

      for (j = 0; j < dE; j++) {
        for (k = 0; k < dim; k++) {
          J[j*dE + k] = g->J[i*dE*dim + j*dim + k];
        }
      }
      switch (dim) {
      case 0:
        for (j = 0; j < dE; j++) {
          for (k = 0; k < dE; k++) {
            J[j * dE + k] = (j == k) ? 1. : 0.;
          }
        }
        break;
      case 1:
        if (dE == 2) {
          PetscReal norm = PetscSqrtReal(J[0] * J[0] + J[2] * J[2]);

          J[1] = -J[2] / norm;
          J[3] =  J[0] / norm;
        } else {
          PetscReal inorm = 1./PetscSqrtReal(J[0] * J[0] + J[3] * J[3] + J[6] * J[6]);
          PetscReal x = J[0] * inorm;
          PetscReal y = J[3] * inorm;
          PetscReal z = J[6] * inorm;

          if (x > 0.) {
            PetscReal inv1pX   = 1./ (1. + x);

            J[1] = -y;              J[2] = -z;
            J[4] = 1. - y*y*inv1pX; J[5] =     -y*z*inv1pX;
            J[7] =     -y*z*inv1pX; J[8] = 1. - z*z*inv1pX;
          } else {
            PetscReal inv1mX   = 1./ (1. - x);

            J[1] = z;               J[2] = y;
            J[4] =     -y*z*inv1mX; J[5] = 1. - y*y*inv1mX;
            J[7] = 1. - z*z*inv1mX; J[8] =     -y*z*inv1mX;
          }
        }
        break;
      case 2:
        {
          PetscReal inorm;

          J[2] = J[3] * J[7] - J[6] * J[4];
          J[5] = J[6] * J[1] - J[0] * J[7];
          J[8] = J[0] * J[4] - J[3] * J[1];

          inorm = 1./ PetscSqrtReal(J[2]*J[2] + J[5]*J[5] + J[8]*J[8]);

          J[2] *= inorm;
          J[5] *= inorm;
          J[8] *= inorm;
        }
        break;
      }
      for (j = 0; j < dE*dE; j++) {
        g->J[i*dE*dE + j] = J[j];
      }
    }
  }
  ierr = PetscFEGeomComplete(g);CHKERRQ(ierr);
  ierr = DMFieldGetDegree(field,pointIS,NULL,&maxDegree);CHKERRQ(ierr);
  g->isAffine = (maxDegree <= 1) ? PETSC_TRUE : PETSC_FALSE;
  if (faceData) {
    if (!field->ops->computeFaceData) SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "DMField implementation does not compute face data\n");
    ierr = (*field->ops->computeFaceData) (field, pointIS, quad, g);CHKERRQ(ierr);
  }
  *geom = g;
  PetscFunctionReturn(0);
}
