#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformSetUp_1D(DMPlexTransform tr)
{
  DM       dm;
  DMLabel  active;
  PetscInt pStart, pEnd, p;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMPlexTransformGetActive(tr, &active));
  PetscCheck(active, PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_WRONGSTATE, "DMPlexTransform must have an adaptation label in order to use 1D algorithm");
  /* Calculate refineType for each cell */
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  for (p = pStart; p < pEnd; ++p) {
    DMLabel        trType = tr->trType;
    DMPolytopeType ct;
    PetscInt       val;

    PetscCall(DMPlexGetCellType(dm, p, &ct));
    switch (ct) {
    case DM_POLYTOPE_POINT:
      PetscCall(DMLabelSetValue(trType, p, 0));
      break;
    case DM_POLYTOPE_SEGMENT:
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
      PetscCall(DMLabelGetValue(active, p, &val));
      if (val == 1) PetscCall(DMLabelSetValue(trType, p, val));
      else PetscCall(DMLabelSetValue(trType, p, 2));
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Cannot handle points of type %s", DMPolytopeTypes[ct]);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_1D(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  PetscInt rt;

  PetscFunctionBeginHot;
  PetscCall(DMLabelGetValue(tr->trType, sp, &rt));
  *rnew = r;
  *onew = o;
  switch (rt) {
  case 1:
    PetscCall(DMPlexTransformGetSubcellOrientation_Regular(tr, sct, sp, so, tct, r, o, rnew, onew));
    break;
  default:
    PetscCall(DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCellTransform_1D(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMLabel  trType = tr->trType;
  PetscInt val;

  PetscFunctionBeginHot;
  PetscCheck(p >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Point argument is invalid");
  PetscCall(DMLabelGetValue(trType, p, &val));
  if (rt) *rt = val;
  switch (source) {
  case DM_POLYTOPE_POINT:
    PetscCall(DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt));
    break;
  case DM_POLYTOPE_POINT_PRISM_TENSOR:
  case DM_POLYTOPE_SEGMENT:
    if (val == 1) PetscCall(DMPlexTransformCellRefine_Regular(tr, source, p, NULL, Nt, target, size, cone, ornt));
    else PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
    break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No refinement strategy for %s", DMPolytopeTypes[source]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetFromOptions_1D(DMPlexTransform tr, PetscOptionItems *PetscOptionsObject)
{
  PetscInt  cells[256], n = 256, i;
  PetscBool flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMPlex Options");
  PetscCall(PetscOptionsIntArray("-dm_plex_transform_1d_ref_cell", "Mark cells for refinement", "", cells, &n, &flg));
  if (flg) {
    DMLabel active;

    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Adaptation Label", &active));
    for (i = 0; i < n; ++i) PetscCall(DMLabelSetValue(active, cells[i], DM_ADAPT_REFINE));
    PetscCall(DMPlexTransformSetActive(tr, active));
    PetscCall(DMLabelDestroy(&active));
  }
  PetscOptionsHeadEnd();
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformView_1D(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    PetscViewerFormat format;
    const char       *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "1D refinement %s\n", name ? name : ""));
    PetscCall(PetscViewerGetFormat(viewer, &format));
    if (format == PETSC_VIEWER_ASCII_INFO_DETAIL) PetscCall(DMLabelView(tr->trType, viewer));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_1D(DMPlexTransform tr)
{
  PetscFunctionBegin;
  PetscCall(PetscFree(tr->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformInitialize_1D(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                  = DMPlexTransformView_1D;
  tr->ops->setfromoptions        = DMPlexTransformSetFromOptions_1D;
  tr->ops->setup                 = DMPlexTransformSetUp_1D;
  tr->ops->destroy               = DMPlexTransformDestroy_1D;
  tr->ops->celltransform         = DMPlexTransformCellTransform_1D;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_1D;
  tr->ops->mapcoordinates        = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_1D(DMPlexTransform tr)
{
  DMPlexRefine_1D *f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&f));
  tr->data = f;

  PetscCall(DMPlexTransformInitialize_1D(tr));
  PetscFunctionReturn(0);
}
