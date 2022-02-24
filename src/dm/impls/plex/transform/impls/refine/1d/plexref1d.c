#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformSetUp_1D(DMPlexTransform tr)
{
  DM             dm;
  DMLabel        active;
  PetscInt       pStart, pEnd, p;

  PetscFunctionBegin;
  CHKERRQ(DMPlexTransformGetDM(tr, &dm));
  CHKERRQ(DMPlexTransformGetActive(tr, &active));
  PetscCheckFalse(!active,PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_WRONGSTATE, "DMPlexTransform must have an adaptation label in order to use 1D algorithm");
  /* Calculate refineType for each cell */
  CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType));
  CHKERRQ(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    DMLabel        trType = tr->trType;
    DMPolytopeType ct;
    PetscInt       val;

    CHKERRQ(DMPlexGetCellType(dm, p, &ct));
    switch (ct) {
      case DM_POLYTOPE_POINT: CHKERRQ(DMLabelSetValue(trType, p, 0)); break;
      case DM_POLYTOPE_SEGMENT:
      case DM_POLYTOPE_POINT_PRISM_TENSOR:
        CHKERRQ(DMLabelGetValue(active, p, &val));
        if (val == 1) CHKERRQ(DMLabelSetValue(trType, p, val));
        else          CHKERRQ(DMLabelSetValue(trType, p, 2));
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle points of type %s", DMPolytopeTypes[ct]);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_1D(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  PetscInt       rt;

  PetscFunctionBeginHot;
  CHKERRQ(DMLabelGetValue(tr->trType, sp, &rt));
  *rnew = r; *onew = o;
  switch (rt) {
    case 1:
      CHKERRQ(DMPlexTransformGetSubcellOrientation_Regular(tr, sct, sp, so, tct, r, o, rnew, onew));
      break;
    default: CHKERRQ(DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCellTransform_1D(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMLabel        trType = tr->trType;
  PetscInt       val;

  PetscFunctionBeginHot;
  PetscCheckFalse(p < 0,PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point argument is invalid");
  CHKERRQ(DMLabelGetValue(trType, p, &val));
  if (rt) *rt = val;
  switch (source) {
    case DM_POLYTOPE_POINT:
      CHKERRQ(DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt));
      break;
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
    case DM_POLYTOPE_SEGMENT:
      if (val == 1) CHKERRQ(DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt));
      else          CHKERRQ(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
      break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No refinement strategy for %s", DMPolytopeTypes[source]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetFromOptions_1D(PetscOptionItems *PetscOptionsObject, DMPlexTransform tr)
{
  PetscInt       cells[256], n = 256, i;
  PetscBool      flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 2);
  CHKERRQ(PetscOptionsHead(PetscOptionsObject,"DMPlex Options"));
  CHKERRQ(PetscOptionsIntArray("-dm_plex_transform_1d_ref_cell", "Mark cells for refinement", "", cells, &n, &flg));
  if (flg) {
    DMLabel active;

    CHKERRQ(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &active));
    for (i = 0; i < n; ++i) CHKERRQ(DMLabelSetValue(active, cells[i], DM_ADAPT_REFINE));
    CHKERRQ(DMPlexTransformSetActive(tr, active));
    CHKERRQ(DMLabelDestroy(&active));
  }
  CHKERRQ(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformView_1D(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscViewerFormat format;
    const char       *name;

    CHKERRQ(PetscObjectGetName((PetscObject) tr, &name));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "1D refinement %s\n", name ? name : ""));
    CHKERRQ(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) {
      CHKERRQ(DMLabelView(tr->trType, viewer));
    }
  } else {
    SETERRQ(PetscObjectComm((PetscObject) tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject) viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_1D(DMPlexTransform tr)
{
  PetscFunctionBegin;
  CHKERRQ(PetscFree(tr->data));
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

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  CHKERRQ(PetscNewLog(tr, &f));
  tr->data = f;

  CHKERRQ(DMPlexTransformInitialize_1D(tr));
  PetscFunctionReturn(0);
}
