#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petsc/private/dtimpl.h> /*I "petscdt.h" I*/

static PetscErrorCode PetscSpacePointView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;
  PetscViewerFormat format;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    ierr = PetscViewerASCIIPrintf(viewer, "Point space in dimension %d:\n", sp->Nv);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = PetscQuadratureView(pt->quad, viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  } else {
    ierr = PetscViewerASCIIPrintf(viewer, "Point space in dimension %d on %d points\n", sp->Nv, pt->quad->numPoints);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Point(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii);CHKERRQ(ierr);
  if (iascii) {ierr = PetscSpacePointView_Ascii(sp, viewer);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Point(PetscSpace sp)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (!pt->quad->points && sp->degree >= 0) {
    ierr = PetscQuadratureDestroy(&pt->quad);CHKERRQ(ierr);
    ierr = PetscDTStroudConicalQuadrature(sp->Nv, sp->Nc, PetscMax(sp->degree + 1, 1), -1.0, 1.0, &pt->quad);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Point(PetscSpace sp)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  ierr = PetscQuadratureDestroy(&pt->quad);CHKERRQ(ierr);
  ierr = PetscFree(pt);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceGetDimension_Point(PetscSpace sp, PetscInt *dim)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;

  PetscFunctionBegin;
  *dim = pt->quad->numPoints;
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceEvaluate_Point(PetscSpace sp, PetscInt npoints, const PetscReal points[], PetscReal B[], PetscReal D[], PetscReal H[])
{
  PetscSpace_Point *pt  = (PetscSpace_Point *) sp->data;
  PetscInt          dim = sp->Nv, pdim = pt->quad->numPoints, d, p, i, c;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  if (npoints != pt->quad->numPoints) SETERRQ2(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot evaluate Point space on %d points != %d size", npoints, pt->quad->numPoints);
  ierr = PetscArrayzero(B, npoints*pdim);CHKERRQ(ierr);
  for (p = 0; p < npoints; ++p) {
    for (i = 0; i < pdim; ++i) {
      for (d = 0; d < dim; ++d) {
        if (PetscAbsReal(points[p*dim+d] - pt->quad->points[p*dim+d]) > 1.0e-10) break;
      }
      if (d >= dim) {B[p*pdim+i] = 1.0; break;}
    }
  }
  /* Replicate for other components */
  for (c = 1; c < sp->Nc; ++c) {
    for (p = 0; p < npoints; ++p) {
      for (i = 0; i < pdim; ++i) {
        B[(c*npoints + p)*pdim + i] = B[p*pdim + i];
      }
    }
  }
  if (D) {ierr = PetscArrayzero(D, npoints*pdim*dim);CHKERRQ(ierr);}
  if (H) {ierr = PetscArrayzero(H, npoints*pdim*dim*dim);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceInitialize_Point(PetscSpace sp)
{
  PetscFunctionBegin;
  sp->ops->setfromoptions = NULL;
  sp->ops->setup          = PetscSpaceSetUp_Point;
  sp->ops->view           = PetscSpaceView_Point;
  sp->ops->destroy        = PetscSpaceDestroy_Point;
  sp->ops->getdimension   = PetscSpaceGetDimension_Point;
  sp->ops->evaluate       = PetscSpaceEvaluate_Point;
  PetscFunctionReturn(0);
}

/*MC
  PETSCSPACEPOINT = "point" - A PetscSpace object that encapsulates functions defined on a set of quadrature points.

  Level: intermediate

.seealso: PetscSpaceType, PetscSpaceCreate(), PetscSpaceSetType()
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Point(PetscSpace sp)
{
  PetscSpace_Point *pt;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  ierr     = PetscNewLog(sp,&pt);CHKERRQ(ierr);
  sp->data = pt;

  sp->Nv = 0;
  sp->maxDegree = PETSC_MAX_INT;
  ierr = PetscQuadratureCreate(PETSC_COMM_SELF, &pt->quad);CHKERRQ(ierr);
  ierr = PetscQuadratureSetData(pt->quad, 0, 1, 0, NULL, NULL);CHKERRQ(ierr);

  ierr = PetscSpaceInitialize_Point(sp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePointSetPoints - Sets the evaluation points for the space to coincide with the points of a quadrature rule

  Logically collective

  Input Parameters:
+ sp - The PetscSpace
- q  - The PetscQuadrature defining the points

  Level: intermediate

.seealso: PetscSpaceCreate(), PetscSpaceSetType()
@*/
PetscErrorCode PetscSpacePointSetPoints(PetscSpace sp, PetscQuadrature q)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;
  PetscErrorCode    ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(q, PETSCQUADRATURE_CLASSID, 2);
  ierr = PetscQuadratureDestroy(&pt->quad);CHKERRQ(ierr);
  ierr = PetscQuadratureDuplicate(q, &pt->quad);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePointGetPoints - Gets the evaluation points for the space as the points of a quadrature rule

  Logically collective

  Input Parameter:
. sp - The PetscSpace

  Output Parameter:
. q  - The PetscQuadrature defining the points

  Level: intermediate

.seealso: PetscSpaceCreate(), PetscSpaceSetType()
@*/
PetscErrorCode PetscSpacePointGetPoints(PetscSpace sp, PetscQuadrature *q)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidPointer(q, 2);
  *q = pt->quad;
  PetscFunctionReturn(0);
}
