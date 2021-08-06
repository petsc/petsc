#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformView_Filter(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    const char *name;

    ierr = PetscObjectGetName((PetscObject) tr, &name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "Filter transformation %s\n", name ? name : "");CHKERRQ(ierr);
  } else {
    SETERRQ1(PetscObjectComm((PetscObject) tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject) viewer)->type_name);
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
  DMPlexTransform_Filter *f = (DMPlexTransform_Filter *) tr->data;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  ierr = DMLabelDestroy(&f->label);CHKERRQ(ierr);
  ierr = PetscFree(f);CHKERRQ(ierr);
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
  tr->ops->view    = DMPlexTransformView_Filter;
  tr->ops->setup   = DMPlexTransformSetUp_Filter;
  tr->ops->destroy = DMPlexTransformDestroy_Filter;
  tr->ops->celltransform = DMPlexTransformCellTransform_Filter;
  tr->ops->mapcoordinates = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Filter(DMPlexTransform tr)
{
  DMPlexTransform_Filter *f;
  PetscErrorCode          ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ierr = PetscNewLog(tr, &f);CHKERRQ(ierr);
  tr->data = f;

  ierr = DMPlexTransformInitialize_Filter(tr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
