#include <petsc/private/petscfeimpl.h> /*I "petscfe.h" I*/
#include <petsc/private/dtimpl.h> /*I "petscdt.h" I*/

static PetscErrorCode PetscSpacePointView_Ascii(PetscSpace sp, PetscViewer viewer)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;
  PetscViewerFormat format;

  PetscFunctionBegin;
  PetscCall(PetscViewerGetFormat(viewer, &format));
  if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Point space in dimension %" PetscInt_FMT ":\n", sp->Nv));
    PetscCall(PetscViewerASCIIPushTab(viewer));
    PetscCall(PetscQuadratureView(pt->quad, viewer));
    PetscCall(PetscViewerASCIIPopTab(viewer));
  } else {
    PetscCall(PetscViewerASCIIPrintf(viewer, "Point space in dimension %" PetscInt_FMT " on %" PetscInt_FMT " points\n", sp->Nv, pt->quad->numPoints));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceView_Point(PetscSpace sp, PetscViewer viewer)
{
  PetscBool      iascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &iascii));
  if (iascii) PetscCall(PetscSpacePointView_Ascii(sp, viewer));
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceSetUp_Point(PetscSpace sp)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;

  PetscFunctionBegin;
  if (!pt->quad->points && sp->degree >= 0) {
    PetscCall(PetscQuadratureDestroy(&pt->quad));
    PetscCall(PetscDTStroudConicalQuadrature(sp->Nv, sp->Nc, PetscMax(sp->degree + 1, 1), -1.0, 1.0, &pt->quad));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PetscSpaceDestroy_Point(PetscSpace sp)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;

  PetscFunctionBegin;
  PetscCall(PetscQuadratureDestroy(&pt->quad));
  PetscCall(PetscFree(pt));
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

  PetscFunctionBegin;
  PetscCheck(npoints == pt->quad->numPoints,PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot evaluate Point space on %" PetscInt_FMT " points != %" PetscInt_FMT " size", npoints, pt->quad->numPoints);
  PetscCall(PetscArrayzero(B, npoints*pdim));
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
  if (D) PetscCall(PetscArrayzero(D, npoints*pdim*dim));
  if (H) PetscCall(PetscArrayzero(H, npoints*pdim*dim*dim));
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

.seealso: `PetscSpaceType`, `PetscSpaceCreate()`, `PetscSpaceSetType()`
M*/

PETSC_EXTERN PetscErrorCode PetscSpaceCreate_Point(PetscSpace sp)
{
  PetscSpace_Point *pt;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscCall(PetscNewLog(sp,&pt));
  sp->data = pt;

  sp->Nv = 0;
  sp->maxDegree = PETSC_MAX_INT;
  PetscCall(PetscQuadratureCreate(PETSC_COMM_SELF, &pt->quad));
  PetscCall(PetscQuadratureSetData(pt->quad, 0, 1, 0, NULL, NULL));

  PetscCall(PetscSpaceInitialize_Point(sp));
  PetscFunctionReturn(0);
}

/*@
  PetscSpacePointSetPoints - Sets the evaluation points for the space to coincide with the points of a quadrature rule

  Logically collective

  Input Parameters:
+ sp - The PetscSpace
- q  - The PetscQuadrature defining the points

  Level: intermediate

.seealso: `PetscSpaceCreate()`, `PetscSpaceSetType()`
@*/
PetscErrorCode PetscSpacePointSetPoints(PetscSpace sp, PetscQuadrature q)
{
  PetscSpace_Point *pt = (PetscSpace_Point *) sp->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(sp, PETSCSPACE_CLASSID, 1);
  PetscValidHeaderSpecific(q, PETSCQUADRATURE_CLASSID, 2);
  PetscCall(PetscQuadratureDestroy(&pt->quad));
  PetscCall(PetscQuadratureDuplicate(q, &pt->quad));
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

.seealso: `PetscSpaceCreate()`, `PetscSpaceSetType()`
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
