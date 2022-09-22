#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformView_Filter(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Filter transformation %s\n", name ? name : ""));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetUp_Filter(DMPlexTransform tr)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_Filter(DMPlexTransform tr)
{
  DMPlexTransform_Filter *f = (DMPlexTransform_Filter *)tr->data;

  PetscFunctionBegin;
  PetscCall(DMLabelDestroy(&f->label));
  PetscCall(PetscFree(f));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCellTransform_Filter(DMPlexTransform cr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformInitialize_Filter(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view           = DMPlexTransformView_Filter;
  tr->ops->setup          = DMPlexTransformSetUp_Filter;
  tr->ops->destroy        = DMPlexTransformDestroy_Filter;
  tr->ops->celltransform  = DMPlexTransformCellTransform_Filter;
  tr->ops->mapcoordinates = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Filter(DMPlexTransform tr)
{
  DMPlexTransform_Filter *f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&f));
  tr->data = f;

  PetscCall(DMPlexTransformInitialize_Filter(tr));
  PetscFunctionReturn(0);
}
