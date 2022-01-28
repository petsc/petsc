#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformSetUp_1D(DMPlexTransform tr)
{
  DM             dm;
  DMLabel        active;
  PetscInt       pStart, pEnd, p;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexTransformGetDM(tr, &dm);CHKERRQ(ierr);
  ierr = DMPlexTransformGetActive(tr, &active);CHKERRQ(ierr);
  PetscAssertFalse(!active,PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_WRONGSTATE, "DMPlexTransform must have an adaptation label in order to use 1D algorithm");
  /* Calculate refineType for each cell */
  ierr = DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType);CHKERRQ(ierr);
  ierr = DMPlexGetChart(dm, &pStart, &pEnd);CHKERRQ(ierr);
  for (p = pStart; p < pEnd; ++p) {
    DMLabel        trType = tr->trType;
    DMPolytopeType ct;
    PetscInt       val;

    ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
    switch (ct) {
      case DM_POLYTOPE_POINT: ierr = DMLabelSetValue(trType, p, 0);CHKERRQ(ierr); break;
      case DM_POLYTOPE_SEGMENT:
      case DM_POLYTOPE_POINT_PRISM_TENSOR:
        ierr = DMLabelGetValue(active, p, &val);CHKERRQ(ierr);
        if (val == 1) {ierr = DMLabelSetValue(trType, p, val);CHKERRQ(ierr);}
        else          {ierr = DMLabelSetValue(trType, p, 2);CHKERRQ(ierr);}
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle points of type %s", DMPolytopeTypes[ct]);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_1D(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  PetscInt       rt;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  ierr = DMLabelGetValue(tr->trType, sp, &rt);CHKERRQ(ierr);
  *rnew = r; *onew = o;
  switch (rt) {
    case 1:
      ierr = DMPlexTransformGetSubcellOrientation_Regular(tr, sct, sp, so, tct, r, o, rnew, onew);CHKERRQ(ierr);
      break;
    default: ierr = DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCellTransform_1D(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMLabel        trType = tr->trType;
  PetscInt       val;
  PetscErrorCode ierr;

  PetscFunctionBeginHot;
  PetscAssertFalse(p < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point argument is invalid");
  ierr = DMLabelGetValue(trType, p, &val);CHKERRQ(ierr);
  if (rt) *rt = val;
  switch (source) {
    case DM_POLYTOPE_POINT:
      ierr = DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt);CHKERRQ(ierr);
      break;
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
    case DM_POLYTOPE_SEGMENT:
      if (val == 1) {ierr = DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt);CHKERRQ(ierr);}
      else          {ierr = DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt);CHKERRQ(ierr);}
      break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No refinement strategy for %s", DMPolytopeTypes[source]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetFromOptions_1D(PetscOptionItems *PetscOptionsObject, DMPlexTransform tr)
{
  PetscInt       cells[256], n = 256, i;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 2);
  ierr = PetscOptionsHead(PetscOptionsObject,"DMPlex Options");CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-dm_plex_transform_1d_ref_cell", "Mark cells for refinement", "", cells, &n, &flg);CHKERRQ(ierr);
  if (flg) {
    DMLabel active;

    ierr = DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &active);CHKERRQ(ierr);
    for (i = 0; i < n; ++i) {ierr = DMLabelSetValue(active, cells[i], DM_ADAPT_REFINE);CHKERRQ(ierr);}
    ierr = DMPlexTransformSetActive(tr, active);CHKERRQ(ierr);
    ierr = DMLabelDestroy(&active);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformView_1D(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool      isascii;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  ierr = PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii);CHKERRQ(ierr);
  if (isascii) {
    PetscViewerFormat format;
    const char       *name;

    ierr = PetscObjectGetName((PetscObject) tr, &name);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer, "1D refinement %s\n", name ? name : "");CHKERRQ(ierr);
    ierr = PetscViewerGetFormat(viewer, &format);CHKERRQ(ierr);
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      ierr = DMLabelView(tr->trType, viewer);CHKERRQ(ierr);
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject) tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject) viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_1D(DMPlexTransform tr)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscFree(tr->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformInitialize_1D(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view           = DMPlexTransformView_1D;
  tr->ops->setfromoptions = DMPlexTransformSetFromOptions_1D;
  tr->ops->setup          = DMPlexTransformSetUp_1D;
  tr->ops->destroy        = DMPlexTransformDestroy_1D;
  tr->ops->celltransform  = DMPlexTransformCellTransform_1D;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_1D;
  tr->ops->mapcoordinates = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_1D(DMPlexTransform tr)
{
  DMPlexRefine_1D *f;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ierr = PetscNewLog(tr, &f);CHKERRQ(ierr);
  tr->data = f;

  ierr = DMPlexTransformInitialize_1D(tr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
