#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

const char *const PlexNormalAlgs[] = {"default", "input", "compute", "compute_bd"};

static PetscErrorCode DMPlexTransformView_Extrude(DMPlexTransform tr, PetscViewer viewer)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;
  PetscBool                isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    DM          dm;
    DMLabel     active;
    PetscInt    dim;
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(DMPlexTransformGetDM(tr, &dm));
    PetscCall(DMGetDimension(dm, &dim));
    PetscCall(DMPlexTransformGetActive(tr, &active));

    PetscCall(PetscViewerASCIIPrintf(viewer, "Extrusion transformation %s\n", name ? name : ""));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  number of layers: %" PetscInt_FMT "\n", ex->layers));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  create tensor cells: %s\n", ex->useTensor ? "YES" : "NO"));
    if (ex->periodic) PetscCall(PetscViewerASCIIPrintf(viewer, "  periodic\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  normal algorithm: %s\n", PlexNormalAlgs[ex->normalAlg]));
    if (ex->normalFunc) PetscCall(PetscViewerASCIIPrintf(viewer, "  normal modified by user function\n"));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetFromOptions_Extrude(DMPlexTransform tr, PetscOptionItems PetscOptionsObject)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;
  PetscReal                th, normal[3], *thicknesses;
  PetscInt                 nl, Nc;
  PetscBool                tensor, sym, per, flg;
  char                     funcname[PETSC_MAX_PATH_LEN];

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMPlexTransform Extrusion Options");
  PetscCall(PetscOptionsBoundedInt("-dm_plex_transform_extrude_layers", "Number of layers to extrude", "", ex->layers, &nl, &flg, 1));
  if (flg) PetscCall(DMPlexTransformExtrudeSetLayers(tr, nl));
  PetscCall(PetscOptionsReal("-dm_plex_transform_extrude_thickness", "Total thickness of extruded layers", "", ex->thickness, &th, &flg));
  if (flg) PetscCall(DMPlexTransformExtrudeSetThickness(tr, th));
  PetscCall(PetscOptionsBool("-dm_plex_transform_extrude_use_tensor", "Create tensor cells", "", ex->useTensor, &tensor, &flg));
  if (flg) PetscCall(DMPlexTransformExtrudeSetTensor(tr, tensor));
  PetscCall(PetscOptionsBool("-dm_plex_transform_extrude_symmetric", "Extrude layers symmetrically about the surface", "", ex->symmetric, &sym, &flg));
  if (flg) PetscCall(DMPlexTransformExtrudeSetSymmetric(tr, sym));
  PetscCall(PetscOptionsBool("-dm_plex_transform_extrude_periodic", "Extrude layers periodically about the surface", "", ex->periodic, &per, &flg));
  if (flg) PetscCall(DMPlexTransformExtrudeSetPeriodic(tr, per));
  Nc = 3;
  PetscCall(PetscOptionsRealArray("-dm_plex_transform_extrude_normal", "Input normal vector for extrusion", "DMPlexTransformExtrudeSetNormal", normal, &Nc, &flg));
  if (flg) {
    // Extrusion dimension might not yet be determined
    PetscCheck(!ex->cdimEx || Nc == ex->cdimEx, PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_SIZ, "Input normal has size %" PetscInt_FMT " != %" PetscInt_FMT " extruded coordinate dimension", Nc, ex->cdimEx);
    PetscCall(DMPlexTransformExtrudeSetNormal(tr, normal));
  }
  PetscCall(PetscOptionsString("-dm_plex_transform_extrude_normal_function", "Function to determine normal vector", "DMPlexTransformExtrudeSetNormalFunction", NULL, funcname, sizeof(funcname), &flg));
  if (flg) {
    PetscSimplePointFn *normalFunc;

    PetscCall(PetscDLSym(NULL, funcname, (void **)&normalFunc));
    PetscCall(DMPlexTransformExtrudeSetNormalFunction(tr, normalFunc));
  }
  nl = ex->layers;
  PetscCall(PetscMalloc1(nl, &thicknesses));
  PetscCall(PetscOptionsRealArray("-dm_plex_transform_extrude_thicknesses", "Thickness of each individual extruded layer", "", thicknesses, &nl, &flg));
  if (flg) {
    PetscCheck(nl, PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_OUTOFRANGE, "Must give at least one thickness for -dm_plex_transform_extrude_thicknesses");
    PetscCall(DMPlexTransformExtrudeSetThicknesses(tr, nl, thicknesses));
  }
  PetscCall(PetscFree(thicknesses));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Determine the implicit dimension pre-extrusion (either the implicit dimension of the DM or of a point in the active set for the transform).
   If that dimension is the same as the current coordinate dimension (ex->dim), the extruded mesh will have a coordinate dimension one greater;
   Otherwise the coordinate dimension will be kept. */
static PetscErrorCode DMPlexTransformExtrudeComputeExtrusionDim(DMPlexTransform tr)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;
  DM                       dm;
  DMLabel                  active;
  PetscInt                 dim, dimExtPoint, dimExtPointG;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexTransformGetActive(tr, &active));
  if (active) {
    IS              valueIS, pointIS;
    const PetscInt *values, *points;
    DMPolytopeType  ct;
    PetscInt        Nv, Np;

    dimExtPoint = 0;
    PetscCall(DMLabelGetValueIS(active, &valueIS));
    PetscCall(ISGetLocalSize(valueIS, &Nv));
    PetscCall(ISGetIndices(valueIS, &values));
    for (PetscInt v = 0; v < Nv; ++v) {
      PetscCall(DMLabelGetStratumIS(active, values[v], &pointIS));
      PetscCall(ISGetLocalSize(pointIS, &Np));
      PetscCall(ISGetIndices(pointIS, &points));
      for (PetscInt p = 0; p < Np; ++p) {
        PetscCall(DMPlexGetCellType(dm, points[p], &ct));
        dimExtPoint = PetscMax(dimExtPoint, DMPolytopeTypeGetDim(ct));
      }
      PetscCall(ISRestoreIndices(pointIS, &points));
      PetscCall(ISDestroy(&pointIS));
    }
    PetscCall(ISRestoreIndices(valueIS, &values));
    PetscCall(ISDestroy(&valueIS));
  } else dimExtPoint = dim;
  PetscCallMPI(MPIU_Allreduce(&dimExtPoint, &dimExtPointG, 1, MPIU_INT, MPI_MAX, PetscObjectComm((PetscObject)tr)));
  ex->dimEx  = PetscMax(dim, dimExtPointG + 1);
  ex->cdimEx = ex->cdim == dimExtPointG ? ex->cdim + 1 : ex->cdim;
  PetscCheck(ex->dimEx <= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Topological dimension for extruded mesh %" PetscInt_FMT " must not exceed 3", ex->dimEx);
  PetscCheck(ex->cdimEx <= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Coordinate dimension for extruded mesh %" PetscInt_FMT " must not exceed 3", ex->cdimEx);
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetDimensions_Extrude(DMPlexTransform tr, DM dm, DM tdm)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;
  PetscInt                 dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSetDimension(tdm, ex->dimEx));
  PetscCall(DMSetCoordinateDim(tdm, ex->cdimEx));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DM_POLYTOPE_POINT produces
     Nl+1 points, or Nl points periodically and
     Nl segments, or tensor segments
*/
static PetscErrorCode DMPlexTransformExtrudeSetUp_Point(DMPlexTransform_Extrude *ex, PetscInt Nl)
{
  const DMPolytopeType ct = DM_POLYTOPE_POINT;
  const PetscInt       Np = ex->periodic ? Nl : Nl + 1;
  PetscInt             Nc, No;

  PetscFunctionBegin;
  ex->Nt[ct] = 2;
  Nc         = 6 * Nl;
  No         = 2 * Nl;
  PetscCall(PetscMalloc4(ex->Nt[ct], &ex->target[ct], ex->Nt[ct], &ex->size[ct], Nc, &ex->cone[ct], No, &ex->ornt[ct]));
  ex->target[ct][0] = DM_POLYTOPE_POINT;
  ex->target[ct][1] = ex->useTensor ? DM_POLYTOPE_POINT_PRISM_TENSOR : DM_POLYTOPE_SEGMENT;
  ex->size[ct][0]   = Np;
  ex->size[ct][1]   = Nl;
  /*   cones for segments/tensor segments */
  for (PetscInt i = 0; i < Nl; ++i) {
    ex->cone[ct][6 * i + 0] = DM_POLYTOPE_POINT;
    ex->cone[ct][6 * i + 1] = 0;
    ex->cone[ct][6 * i + 2] = i;
    ex->cone[ct][6 * i + 3] = DM_POLYTOPE_POINT;
    ex->cone[ct][6 * i + 4] = 0;
    ex->cone[ct][6 * i + 5] = (i + 1) % Np;
  }
  for (PetscInt i = 0; i < No; ++i) ex->ornt[ct][i] = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DM_POLYTOPE_SEGMENT produces
     Nl+1 segments, or Nl segments periodically and
     Nl quads, or tensor quads
*/
static PetscErrorCode DMPlexTransformExtrudeSetUp_Segment(DMPlexTransform_Extrude *ex, PetscInt Nl)
{
  const DMPolytopeType ct = DM_POLYTOPE_SEGMENT;
  const PetscInt       Np = ex->periodic ? Nl : Nl + 1;
  PetscInt             Nc, No, coff, ooff;

  PetscFunctionBegin;
  ex->Nt[ct] = 2;
  Nc         = 8 * Np + 14 * Nl;
  No         = 2 * Np + 4 * Nl;
  PetscCall(PetscMalloc4(ex->Nt[ct], &ex->target[ct], ex->Nt[ct], &ex->size[ct], Nc, &ex->cone[ct], No, &ex->ornt[ct]));
  ex->target[ct][0] = DM_POLYTOPE_SEGMENT;
  ex->target[ct][1] = ex->useTensor ? DM_POLYTOPE_SEG_PRISM_TENSOR : DM_POLYTOPE_QUADRILATERAL;
  ex->size[ct][0]   = Np;
  ex->size[ct][1]   = Nl;
  /*   cones for segments */
  for (PetscInt i = 0; i < Np; ++i) {
    ex->cone[ct][8 * i + 0] = DM_POLYTOPE_POINT;
    ex->cone[ct][8 * i + 1] = 1;
    ex->cone[ct][8 * i + 2] = 0;
    ex->cone[ct][8 * i + 3] = i;
    ex->cone[ct][8 * i + 4] = DM_POLYTOPE_POINT;
    ex->cone[ct][8 * i + 5] = 1;
    ex->cone[ct][8 * i + 6] = 1;
    ex->cone[ct][8 * i + 7] = i;
  }
  for (PetscInt i = 0; i < 2 * Np; ++i) ex->ornt[ct][i] = 0;
  /*   cones for quads/tensor quads */
  coff = 8 * Np;
  ooff = 2 * Np;
  for (PetscInt i = 0; i < Nl; ++i) {
    if (ex->useTensor) {
      ex->cone[ct][coff + 14 * i + 0]  = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff + 14 * i + 1]  = 0;
      ex->cone[ct][coff + 14 * i + 2]  = i;
      ex->cone[ct][coff + 14 * i + 3]  = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff + 14 * i + 4]  = 0;
      ex->cone[ct][coff + 14 * i + 5]  = (i + 1) % Np;
      ex->cone[ct][coff + 14 * i + 6]  = DM_POLYTOPE_POINT_PRISM_TENSOR;
      ex->cone[ct][coff + 14 * i + 7]  = 1;
      ex->cone[ct][coff + 14 * i + 8]  = 0;
      ex->cone[ct][coff + 14 * i + 9]  = i;
      ex->cone[ct][coff + 14 * i + 10] = DM_POLYTOPE_POINT_PRISM_TENSOR;
      ex->cone[ct][coff + 14 * i + 11] = 1;
      ex->cone[ct][coff + 14 * i + 12] = 1;
      ex->cone[ct][coff + 14 * i + 13] = i;
      ex->ornt[ct][ooff + 4 * i + 0]   = 0;
      ex->ornt[ct][ooff + 4 * i + 1]   = 0;
      ex->ornt[ct][ooff + 4 * i + 2]   = 0;
      ex->ornt[ct][ooff + 4 * i + 3]   = 0;
    } else {
      ex->cone[ct][coff + 14 * i + 0]  = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff + 14 * i + 1]  = 0;
      ex->cone[ct][coff + 14 * i + 2]  = i;
      ex->cone[ct][coff + 14 * i + 3]  = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff + 14 * i + 4]  = 1;
      ex->cone[ct][coff + 14 * i + 5]  = 1;
      ex->cone[ct][coff + 14 * i + 6]  = i;
      ex->cone[ct][coff + 14 * i + 7]  = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff + 14 * i + 8]  = 0;
      ex->cone[ct][coff + 14 * i + 9]  = (i + 1) % Np;
      ex->cone[ct][coff + 14 * i + 10] = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff + 14 * i + 11] = 1;
      ex->cone[ct][coff + 14 * i + 12] = 0;
      ex->cone[ct][coff + 14 * i + 13] = i;
      ex->ornt[ct][ooff + 4 * i + 0]   = 0;
      ex->ornt[ct][ooff + 4 * i + 1]   = 0;
      ex->ornt[ct][ooff + 4 * i + 2]   = -1;
      ex->ornt[ct][ooff + 4 * i + 3]   = -1;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DM_POLYTOPE_TRIANGLE produces
     Nl+1 triangles, or Nl triangles periodically and
     Nl triangular prisms/tensor triangular prisms
*/
static PetscErrorCode DMPlexTransformExtrudeSetUp_Triangle(DMPlexTransform_Extrude *ex, PetscInt Nl)
{
  const DMPolytopeType ct = DM_POLYTOPE_TRIANGLE;
  const PetscInt       Np = ex->periodic ? Nl : Nl + 1;
  PetscInt             Nc, No, coff, ooff;

  PetscFunctionBegin;
  ex->Nt[ct] = 2;
  Nc         = 12 * Np + 18 * Nl;
  No         = 3 * Np + 5 * Nl;
  PetscCall(PetscMalloc4(ex->Nt[ct], &ex->target[ct], ex->Nt[ct], &ex->size[ct], Nc, &ex->cone[ct], No, &ex->ornt[ct]));
  ex->target[ct][0] = DM_POLYTOPE_TRIANGLE;
  ex->target[ct][1] = ex->useTensor ? DM_POLYTOPE_TRI_PRISM_TENSOR : DM_POLYTOPE_TRI_PRISM;
  ex->size[ct][0]   = Np;
  ex->size[ct][1]   = Nl;
  /*   cones for triangles */
  for (PetscInt i = 0; i < Np; ++i) {
    ex->cone[ct][12 * i + 0]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][12 * i + 1]  = 1;
    ex->cone[ct][12 * i + 2]  = 0;
    ex->cone[ct][12 * i + 3]  = i;
    ex->cone[ct][12 * i + 4]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][12 * i + 5]  = 1;
    ex->cone[ct][12 * i + 6]  = 1;
    ex->cone[ct][12 * i + 7]  = i;
    ex->cone[ct][12 * i + 8]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][12 * i + 9]  = 1;
    ex->cone[ct][12 * i + 10] = 2;
    ex->cone[ct][12 * i + 11] = i;
  }
  for (PetscInt i = 0; i < 3 * Np; ++i) ex->ornt[ct][i] = 0;
  /*   cones for triangular prisms/tensor triangular prisms */
  coff = 12 * Np;
  ooff = 3 * Np;
  for (PetscInt i = 0; i < Nl; ++i) {
    if (ex->useTensor) {
      ex->cone[ct][coff + 18 * i + 0]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[ct][coff + 18 * i + 1]  = 0;
      ex->cone[ct][coff + 18 * i + 2]  = i;
      ex->cone[ct][coff + 18 * i + 3]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[ct][coff + 18 * i + 4]  = 0;
      ex->cone[ct][coff + 18 * i + 5]  = (i + 1) % Np;
      ex->cone[ct][coff + 18 * i + 6]  = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff + 18 * i + 7]  = 1;
      ex->cone[ct][coff + 18 * i + 8]  = 0;
      ex->cone[ct][coff + 18 * i + 9]  = i;
      ex->cone[ct][coff + 18 * i + 10] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff + 18 * i + 11] = 1;
      ex->cone[ct][coff + 18 * i + 12] = 1;
      ex->cone[ct][coff + 18 * i + 13] = i;
      ex->cone[ct][coff + 18 * i + 14] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff + 18 * i + 15] = 1;
      ex->cone[ct][coff + 18 * i + 16] = 2;
      ex->cone[ct][coff + 18 * i + 17] = i;
      ex->ornt[ct][ooff + 5 * i + 0]   = 0;
      ex->ornt[ct][ooff + 5 * i + 1]   = 0;
      ex->ornt[ct][ooff + 5 * i + 2]   = 0;
      ex->ornt[ct][ooff + 5 * i + 3]   = 0;
      ex->ornt[ct][ooff + 5 * i + 4]   = 0;
    } else {
      ex->cone[ct][coff + 18 * i + 0]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[ct][coff + 18 * i + 1]  = 0;
      ex->cone[ct][coff + 18 * i + 2]  = i;
      ex->cone[ct][coff + 18 * i + 3]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[ct][coff + 18 * i + 4]  = 0;
      ex->cone[ct][coff + 18 * i + 5]  = (i + 1) % Np;
      ex->cone[ct][coff + 18 * i + 6]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 18 * i + 7]  = 1;
      ex->cone[ct][coff + 18 * i + 8]  = 0;
      ex->cone[ct][coff + 18 * i + 9]  = i;
      ex->cone[ct][coff + 18 * i + 10] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 18 * i + 11] = 1;
      ex->cone[ct][coff + 18 * i + 12] = 1;
      ex->cone[ct][coff + 18 * i + 13] = i;
      ex->cone[ct][coff + 18 * i + 14] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 18 * i + 15] = 1;
      ex->cone[ct][coff + 18 * i + 16] = 2;
      ex->cone[ct][coff + 18 * i + 17] = i;
      ex->ornt[ct][ooff + 5 * i + 0]   = -2;
      ex->ornt[ct][ooff + 5 * i + 1]   = 0;
      ex->ornt[ct][ooff + 5 * i + 2]   = 0;
      ex->ornt[ct][ooff + 5 * i + 3]   = 0;
      ex->ornt[ct][ooff + 5 * i + 4]   = 0;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DM_POLYTOPE_QUADRILATERAL produces
     Nl+1 quads, or Nl quads periodically and
     Nl hexes/tensor hexes
*/
static PetscErrorCode DMPlexTransformExtrudeSetUp_Quadrilateral(DMPlexTransform_Extrude *ex, PetscInt Nl)
{
  const DMPolytopeType ct = DM_POLYTOPE_QUADRILATERAL;
  const PetscInt       Np = ex->periodic ? Nl : Nl + 1;
  PetscInt             Nc, No, coff, ooff;

  PetscFunctionBegin;
  ex->Nt[ct] = 2;
  Nc         = 16 * Np + 22 * Nl;
  No         = 4 * Np + 6 * Nl;
  PetscCall(PetscMalloc4(ex->Nt[ct], &ex->target[ct], ex->Nt[ct], &ex->size[ct], Nc, &ex->cone[ct], No, &ex->ornt[ct]));
  ex->target[ct][0] = DM_POLYTOPE_QUADRILATERAL;
  ex->target[ct][1] = ex->useTensor ? DM_POLYTOPE_QUAD_PRISM_TENSOR : DM_POLYTOPE_HEXAHEDRON;
  ex->size[ct][0]   = Np;
  ex->size[ct][1]   = Nl;
  /*   cones for quads */
  for (PetscInt i = 0; i < Np; ++i) {
    ex->cone[ct][16 * i + 0]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][16 * i + 1]  = 1;
    ex->cone[ct][16 * i + 2]  = 0;
    ex->cone[ct][16 * i + 3]  = i;
    ex->cone[ct][16 * i + 4]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][16 * i + 5]  = 1;
    ex->cone[ct][16 * i + 6]  = 1;
    ex->cone[ct][16 * i + 7]  = i;
    ex->cone[ct][16 * i + 8]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][16 * i + 9]  = 1;
    ex->cone[ct][16 * i + 10] = 2;
    ex->cone[ct][16 * i + 11] = i;
    ex->cone[ct][16 * i + 12] = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][16 * i + 13] = 1;
    ex->cone[ct][16 * i + 14] = 3;
    ex->cone[ct][16 * i + 15] = i;
  }
  for (PetscInt i = 0; i < 4 * Np; ++i) ex->ornt[ct][i] = 0;
  /*   cones for hexes/tensor hexes */
  coff = 16 * Np;
  ooff = 4 * Np;
  for (PetscInt i = 0; i < Nl; ++i) {
    if (ex->useTensor) {
      ex->cone[ct][coff + 22 * i + 0]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 22 * i + 1]  = 0;
      ex->cone[ct][coff + 22 * i + 2]  = i;
      ex->cone[ct][coff + 22 * i + 3]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 22 * i + 4]  = 0;
      ex->cone[ct][coff + 22 * i + 5]  = (i + 1) % Np;
      ex->cone[ct][coff + 22 * i + 6]  = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff + 22 * i + 7]  = 1;
      ex->cone[ct][coff + 22 * i + 8]  = 0;
      ex->cone[ct][coff + 22 * i + 9]  = i;
      ex->cone[ct][coff + 22 * i + 10] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff + 22 * i + 11] = 1;
      ex->cone[ct][coff + 22 * i + 12] = 1;
      ex->cone[ct][coff + 22 * i + 13] = i;
      ex->cone[ct][coff + 22 * i + 14] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff + 22 * i + 15] = 1;
      ex->cone[ct][coff + 22 * i + 16] = 2;
      ex->cone[ct][coff + 22 * i + 17] = i;
      ex->cone[ct][coff + 22 * i + 18] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff + 22 * i + 19] = 1;
      ex->cone[ct][coff + 22 * i + 20] = 3;
      ex->cone[ct][coff + 22 * i + 21] = i;
      ex->ornt[ct][ooff + 6 * i + 0]   = 0;
      ex->ornt[ct][ooff + 6 * i + 1]   = 0;
      ex->ornt[ct][ooff + 6 * i + 2]   = 0;
      ex->ornt[ct][ooff + 6 * i + 3]   = 0;
      ex->ornt[ct][ooff + 6 * i + 4]   = 0;
      ex->ornt[ct][ooff + 6 * i + 5]   = 0;
    } else {
      ex->cone[ct][coff + 22 * i + 0]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 22 * i + 1]  = 0;
      ex->cone[ct][coff + 22 * i + 2]  = i;
      ex->cone[ct][coff + 22 * i + 3]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 22 * i + 4]  = 0;
      ex->cone[ct][coff + 22 * i + 5]  = (i + 1) % Np;
      ex->cone[ct][coff + 22 * i + 6]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 22 * i + 7]  = 1;
      ex->cone[ct][coff + 22 * i + 8]  = 0;
      ex->cone[ct][coff + 22 * i + 9]  = i;
      ex->cone[ct][coff + 22 * i + 10] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 22 * i + 11] = 1;
      ex->cone[ct][coff + 22 * i + 12] = 2;
      ex->cone[ct][coff + 22 * i + 13] = i;
      ex->cone[ct][coff + 22 * i + 14] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 22 * i + 15] = 1;
      ex->cone[ct][coff + 22 * i + 16] = 1;
      ex->cone[ct][coff + 22 * i + 17] = i;
      ex->cone[ct][coff + 22 * i + 18] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff + 22 * i + 19] = 1;
      ex->cone[ct][coff + 22 * i + 20] = 3;
      ex->cone[ct][coff + 22 * i + 21] = i;
      ex->ornt[ct][ooff + 6 * i + 0]   = -2;
      ex->ornt[ct][ooff + 6 * i + 1]   = 0;
      ex->ornt[ct][ooff + 6 * i + 2]   = 0;
      ex->ornt[ct][ooff + 6 * i + 3]   = 0;
      ex->ornt[ct][ooff + 6 * i + 4]   = 0;
      ex->ornt[ct][ooff + 6 * i + 5]   = 1;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The refine types for extrusion are:

  ct:       For any normally extruded point
  ct + 100: For any point which should just return itself
*/
static PetscErrorCode DMPlexTransformSetUp_Extrude(DMPlexTransform tr)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;
  DM                       dm;
  DMLabel                  active;
  PetscInt                 Nl = ex->layers, l, ict, dim;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformExtrudeComputeExtrusionDim(tr));
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexTransformGetActive(tr, &active));
  if (active) {
    DMLabel  celltype;
    PetscInt pStart, pEnd, p;

    PetscCall(DMPlexGetCellTypeLabel(dm, &celltype));
    PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType));
    PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
    for (p = pStart; p < pEnd; ++p) {
      PetscInt ct, val;

      PetscCall(DMLabelGetValue(celltype, p, &ct));
      PetscCall(DMLabelGetValue(active, p, &val));
      if (val < 0) {
        PetscCall(DMLabelSetValue(tr->trType, p, ct + 100));
      } else {
        PetscCall(DMLabelSetValue(tr->trType, p, ct));
      }
    }
  }
  if (ex->normalAlg != NORMAL_INPUT) {
    if (dim != ex->cdim) ex->normalAlg = NORMAL_COMPUTE;
    else if (active) ex->normalAlg = NORMAL_COMPUTE_BD;
  }
  // Need this to determine face sharing
  PetscMPIInt size;

  PetscCallMPI(MPI_Comm_size(PetscObjectComm((PetscObject)dm), &size));
  if (size > 1) {
    PetscSF  sf;
    PetscInt Nr;

    PetscCall(DMGetPointSF(dm, &sf));
    PetscCall(PetscSFGetGraph(sf, &Nr, NULL, NULL, NULL));
    if (Nr >= 0) {
      PetscCall(PetscSFComputeDegreeBegin(sf, &ex->degree));
      PetscCall(PetscSFComputeDegreeEnd(sf, &ex->degree));
    }
  }
  // Create normal field
  if (ex->normalAlg == NORMAL_COMPUTE_BD) {
    PetscSection s;
    PetscInt     vStart, vEnd;

    PetscMPIInt rank;
    PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

    PetscCall(DMClone(dm, &ex->dmNormal));
    PetscCall(DMGetLocalSection(ex->dmNormal, &s));
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    PetscCall(PetscSectionSetNumFields(s, 1));
    PetscCall(PetscSectionSetChart(s, vStart, vEnd));
    for (PetscInt v = vStart; v < vEnd; ++v) PetscCall(PetscSectionSetDof(s, v, ex->cdimEx));
    PetscCall(PetscSectionSetUp(s));
    PetscCall(DMCreateLocalVector(ex->dmNormal, &ex->vecNormal));
    PetscCall(PetscObjectSetName((PetscObject)ex->vecNormal, "Normal Field"));

    // find an active point in the closure of v and use its coordinate normal as the extrusion direction
    PetscSF         sf;
    const PetscInt *leaves;
    PetscScalar    *a, *normal;
    PetscInt        Nl;

    PetscCall(DMGetPointSF(ex->dmNormal, &sf));
    PetscCall(PetscSFGetGraph(sf, NULL, &Nl, &leaves, NULL));
    PetscCall(VecGetArrayWrite(ex->vecNormal, &a));
    for (PetscInt v = vStart; v < vEnd; ++v) {
      PetscInt *star = NULL;
      PetscInt  starSize, pStart, pEnd;

      PetscCall(DMPlexGetDepthStratum(ex->dmNormal, ex->cdimEx - 1, &pStart, &pEnd));
      PetscCall(DMPlexGetTransitiveClosure(ex->dmNormal, v, PETSC_FALSE, &starSize, &star));
      PetscCall(DMPlexPointLocalRef(ex->dmNormal, v, a, &normal));
      for (PetscInt st = 0; st < starSize * 2; st += 2) {
        const PetscInt face = star[st];
        if ((face >= pStart) && (face < pEnd)) {
          PetscReal       cnormal[3] = {0, 0, 0};
          const PetscInt *supp;
          PetscInt        suppSize, floc = -1;
          PetscBool       shared;

          PetscCall(DMPlexComputeCellGeometryFVM(ex->dmNormal, face, NULL, NULL, cnormal));
          PetscCall(DMPlexGetSupportSize(ex->dmNormal, face, &suppSize));
          PetscCall(DMPlexGetSupport(ex->dmNormal, face, &supp));
          // Only use external faces, so I can get the orientation from any cell
          if (leaves) PetscCall(PetscFindInt(face, Nl, leaves, &floc));
          shared = floc >= 0 || (ex->degree && ex->degree[face]) ? PETSC_TRUE : PETSC_FALSE;
          if (suppSize == 1 && !shared) {
            const PetscInt *cone, *ornt;
            PetscInt        coneSize, c;

            PetscCall(DMPlexGetConeSize(ex->dmNormal, supp[0], &coneSize));
            PetscCall(DMPlexGetCone(ex->dmNormal, supp[0], &cone));
            PetscCall(DMPlexGetConeOrientation(ex->dmNormal, supp[0], &ornt));
            for (c = 0; c < coneSize; ++c)
              if (cone[c] == face) break;
            PetscCheck(c < coneSize, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Asymmetry in cone/support");
            if (ornt[c] < 0)
              for (PetscInt d = 0; d < ex->cdimEx; ++d) cnormal[d] *= -1.;
            for (PetscInt d = 0; d < ex->cdimEx; ++d) normal[d] += cnormal[d];
          }
        }
      }
      PetscCall(DMPlexRestoreTransitiveClosure(ex->dmNormal, v, PETSC_FALSE, &starSize, &star));
    }
    PetscCall(VecRestoreArrayWrite(ex->vecNormal, &a));

    Vec g;

    PetscCall(DMGetGlobalVector(ex->dmNormal, &g));
    PetscCall(VecSet(g, 0.));
    PetscCall(DMLocalToGlobal(ex->dmNormal, ex->vecNormal, ADD_VALUES, g));
    PetscCall(DMGlobalToLocal(ex->dmNormal, g, INSERT_VALUES, ex->vecNormal));
    PetscCall(DMRestoreGlobalVector(ex->dmNormal, &g));
  }
  PetscCall(PetscMalloc5(DM_NUM_POLYTOPES, &ex->Nt, DM_NUM_POLYTOPES, &ex->target, DM_NUM_POLYTOPES, &ex->size, DM_NUM_POLYTOPES, &ex->cone, DM_NUM_POLYTOPES, &ex->ornt));
  for (ict = 0; ict < DM_NUM_POLYTOPES; ++ict) {
    ex->Nt[ict]     = -1;
    ex->target[ict] = NULL;
    ex->size[ict]   = NULL;
    ex->cone[ict]   = NULL;
    ex->ornt[ict]   = NULL;
  }
  PetscCall(DMPlexTransformExtrudeSetUp_Point(ex, Nl));
  PetscCall(DMPlexTransformExtrudeSetUp_Segment(ex, Nl));
  PetscCall(DMPlexTransformExtrudeSetUp_Triangle(ex, Nl));
  PetscCall(DMPlexTransformExtrudeSetUp_Quadrilateral(ex, Nl));
  /* Layers positions */
  if (!ex->Nth) {
    if (ex->symmetric)
      for (l = 0; l <= ex->layers; ++l) ex->layerPos[l] = (ex->thickness * l) / ex->layers - ex->thickness / 2;
    else
      for (l = 0; l <= ex->layers; ++l) ex->layerPos[l] = (ex->thickness * l) / ex->layers;
  } else {
    ex->thickness   = 0.;
    ex->layerPos[0] = 0.;
    for (l = 0; l < ex->layers; ++l) {
      const PetscReal t   = ex->thicknesses[PetscMin(l, ex->Nth - 1)];
      ex->layerPos[l + 1] = ex->layerPos[l] + t;
      ex->thickness += t;
    }
    if (ex->symmetric)
      for (l = 0; l <= ex->layers; ++l) ex->layerPos[l] -= ex->thickness / 2.;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformDestroy_Extrude(DMPlexTransform tr)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;
  PetscInt                 ct;

  PetscFunctionBegin;
  if (ex->target) {
    for (ct = 0; ct < DM_NUM_POLYTOPES; ++ct) PetscCall(PetscFree4(ex->target[ct], ex->size[ct], ex->cone[ct], ex->ornt[ct]));
  }
  PetscCall(PetscFree5(ex->Nt, ex->target, ex->size, ex->cone, ex->ornt));
  PetscCall(PetscFree(ex->layerPos));
  PetscCall(DMDestroy(&ex->dmNormal));
  PetscCall(VecDestroy(&ex->vecNormal));
  PetscCall(PetscFree(ex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_Extrude(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  DMPlexTransform_Extrude *ex     = (DMPlexTransform_Extrude *)tr->data;
  DMLabel                  trType = tr->trType, active;
  PetscBool                onBd   = PETSC_FALSE;
  PetscInt                 rt;

  PetscFunctionBeginHot;
  *rnew = r;
  *onew = DMPolytopeTypeComposeOrientation(tct, o, so);
  PetscCall(DMPlexTransformGetActive(tr, &active));
  if (!so && !active) PetscFunctionReturn(PETSC_SUCCESS);
  if (trType) {
    PetscCall(DMLabelGetValue(tr->trType, sp, &rt));
    if (rt >= 100) PetscFunctionReturn(PETSC_SUCCESS);
  }
  if (active) {
    // Get orientation of boundary face in cell
    if (DMPolytopeTypeGetDim(sct) == ex->dimEx - 1) {
      DM              dm;
      const PetscInt *supp, *cone, *ornt;
      PetscInt        suppSize, coneSize, c;

      PetscCall(DMPlexTransformGetDM(tr, &dm));
      PetscCall(DMPlexGetSupportSize(dm, sp, &suppSize));
      PetscCall(DMPlexGetSupport(dm, sp, &supp));
      PetscCheck(suppSize == 1, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Source point %" PetscInt_FMT " is not a boundary face", sp);
      PetscCall(DMPlexGetConeSize(dm, supp[0], &coneSize));
      PetscCall(DMPlexGetOrientedCone(dm, supp[0], &cone, &ornt));
      for (c = 0; c < coneSize; ++c)
        if (cone[c] == sp) break;
      PetscCheck(c < coneSize, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Source point %" PetscInt_FMT " not found in cone of support %" PetscInt_FMT, sp, supp[0]);
      o    = ornt[c];
      onBd = PETSC_TRUE;
      PetscCall(DMPlexRestoreOrientedCone(dm, supp[0], &cone, &ornt));
    }
  }
  if (ex->useTensor) {
    switch (sct) {
    case DM_POLYTOPE_POINT:
      break;
    case DM_POLYTOPE_SEGMENT:
      switch (tct) {
      case DM_POLYTOPE_SEGMENT:
        break;
      case DM_POLYTOPE_SEG_PRISM_TENSOR:
        *onew = onBd ? DMPolytopeTypeComposeOrientation(tct, o, so ? 0 : -1) : DMPolytopeTypeComposeOrientation(tct, o, so ? -1 : 0);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
    // We need to handle identity extrusions from volumes (TET, HEX, etc) when boundary faces are being extruded
    case DM_POLYTOPE_TRIANGLE:
      switch (tct) {
      case DM_POLYTOPE_TRIANGLE:
        break;
      case DM_POLYTOPE_TRI_PRISM_TENSOR:
        *onew = onBd ? DMPolytopeTypeComposeOrientation(tct, o, so ? 0 : -1) : DMPolytopeTypeComposeOrientation(tct, o, so ? -1 : 0);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      switch (tct) {
      case DM_POLYTOPE_QUADRILATERAL:
        break;
      case DM_POLYTOPE_QUAD_PRISM_TENSOR:
        *onew = onBd ? DMPolytopeTypeComposeOrientation(tct, o, so ? 0 : -1) : DMPolytopeTypeComposeOrientation(tct, o, so ? -1 : 0);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell type %s", DMPolytopeTypes[sct]);
    }
  } else {
    switch (sct) {
    case DM_POLYTOPE_POINT:
      break;
    case DM_POLYTOPE_SEGMENT:
      switch (tct) {
      case DM_POLYTOPE_SEGMENT:
        break;
      case DM_POLYTOPE_QUADRILATERAL:
        *onew = onBd ? DMPolytopeTypeComposeOrientation(tct, o, so ? 0 : -3) : DMPolytopeTypeComposeOrientation(tct, o, so ? -3 : 0);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
    case DM_POLYTOPE_TRIANGLE:
      switch (tct) {
      case DM_POLYTOPE_TRIANGLE:
        break;
      case DM_POLYTOPE_TRI_PRISM:
        *onew = onBd ? DMPolytopeTypeComposeOrientation(tct, o, so ? 0 : -1) : DMPolytopeTypeComposeOrientation(tct, o, so ? -1 : 0);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      switch (tct) {
      case DM_POLYTOPE_QUADRILATERAL:
        break;
      case DM_POLYTOPE_HEXAHEDRON:
        *onew = onBd ? DMPolytopeTypeComposeOrientation(tct, o, so ? 0 : -1) : DMPolytopeTypeComposeOrientation(tct, o, so ? -1 : 0);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell type %s", DMPolytopeTypes[sct]);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformCellTransform_Extrude(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMPlexTransform_Extrude *ex     = (DMPlexTransform_Extrude *)tr->data;
  DMLabel                  trType = tr->trType;
  PetscBool                ignore = PETSC_FALSE, identity = PETSC_FALSE;
  PetscInt                 val = 0;

  PetscFunctionBegin;
  if (trType) {
    PetscCall(DMLabelGetValue(trType, p, &val));
    identity = val >= 100 ? PETSC_TRUE : PETSC_FALSE;
  } else {
    ignore = ex->Nt[source] < 0 ? PETSC_TRUE : PETSC_FALSE;
  }
  if (rt) *rt = val;
  if (ignore) {
    /* Ignore cells that cannot be extruded */
    *Nt     = 0;
    *target = NULL;
    *size   = NULL;
    *cone   = NULL;
    *ornt   = NULL;
  } else if (identity) {
    PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
  } else {
    *Nt     = ex->Nt[source];
    *target = ex->target[source];
    *size   = ex->size[source];
    *cone   = ex->cone[source];
    *ornt   = ex->ornt[source];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* Computes new vertex along normal */
static PetscErrorCode DMPlexTransformMapCoordinates_Extrude(DMPlexTransform tr, DMPolytopeType pct, DMPolytopeType ct, PetscInt p, PetscInt r, PetscInt Nv, PetscInt dE, const PetscScalar in[], PetscScalar out[])
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;
  DM                       dm;
  DMLabel                  active;
  PetscReal                ones2[2]  = {0., 1.};
  PetscReal                ones3[3]  = {0., 0., 1.};
  PetscReal                normal[3] = {0., 0., 0.};
  PetscReal                norm      = 0.;
  PetscInt                 dEx       = ex->cdimEx;
  PetscInt                 dim, cStart, cEnd;

  PetscFunctionBeginHot;
  PetscCheck(pct == DM_POLYTOPE_POINT, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for parent point type %s", DMPolytopeTypes[pct]);
  PetscCheck(ct == DM_POLYTOPE_POINT, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for refined point type %s", DMPolytopeTypes[ct]);
  PetscCheck(Nv == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Vertices should be produced from a single vertex, not %" PetscInt_FMT, Nv);
  PetscCheck(dE == ex->cdim, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Coordinate dim %" PetscInt_FMT " != %" PetscInt_FMT " original dimension", dE, ex->cdim);
  PetscCheck(dEx <= 3, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Coordinate dimension for extruded mesh %" PetscInt_FMT " must not exceed 3", dEx);

  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexTransformGetActive(tr, &active));
  switch (ex->normalAlg) {
  case NORMAL_DEFAULT:
    switch (ex->cdimEx) {
    case 2:
      for (PetscInt d = 0; d < dEx; ++d) normal[d] = ones2[d];
      break;
    case 3:
      for (PetscInt d = 0; d < dEx; ++d) normal[d] = ones3[d];
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "No default normal for dimension %" PetscInt_FMT, ex->cdimEx);
    }
    break;
  case NORMAL_INPUT:
    for (PetscInt d = 0; d < dEx; ++d) normal[d] = ex->normal[d];
    break;
  case NORMAL_COMPUTE: {
    PetscInt *star = NULL;
    PetscInt  starSize;

    PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_FALSE, &starSize, &star));
    for (PetscInt st = 0; st < starSize * 2; st += 2) {
      if ((star[st] >= cStart) && (star[st] < cEnd)) {
        PetscReal cnormal[3] = {0, 0, 0};

        PetscCall(DMPlexComputeCellGeometryFVM(dm, star[st], NULL, NULL, cnormal));
        for (PetscInt d = 0; d < dEx; ++d) normal[d] += cnormal[d];
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_FALSE, &starSize, &star));
  } break;
  case NORMAL_COMPUTE_BD: {
    const PetscScalar *a;
    PetscScalar       *vnormal;

    PetscCall(VecGetArrayRead(ex->vecNormal, &a));
    PetscCall(DMPlexPointLocalRead(ex->dmNormal, p, a, (void *)&vnormal));
    for (PetscInt d = 0; d < dEx; ++d) normal[d] = PetscRealPart(vnormal[d]);
    PetscCall(VecRestoreArrayRead(ex->vecNormal, &a));
  } break;
  default:
    SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to determine normal for extrusion");
  }
  if (ex->normalFunc) {
    PetscScalar n[3];
    PetscReal   x[3], dot = 0.;

    for (PetscInt d = 0; d < ex->cdim; ++d) x[d] = PetscRealPart(in[d]);
    PetscCall((*ex->normalFunc)(ex->cdim, 0., x, r, n, NULL));
    for (PetscInt d = 0; d < dEx; ++d) dot += PetscRealPart(n[d]) * normal[d];
    for (PetscInt d = 0; d < dEx; ++d) normal[d] = PetscSign(dot) * PetscRealPart(n[d]);
  }
  for (PetscInt d = 0; d < dEx; ++d) norm += PetscSqr(normal[d]);
  for (PetscInt d = 0; d < dEx; ++d) normal[d] *= norm == 0.0 ? 1.0 : 1. / PetscSqrtReal(norm);
  for (PetscInt d = 0; d < dEx; ++d) out[d] = normal[d] * ex->layerPos[r];
  for (PetscInt d = 0; d < ex->cdim; ++d) out[d] += in[d];
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformInitialize_Extrude(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                  = DMPlexTransformView_Extrude;
  tr->ops->setfromoptions        = DMPlexTransformSetFromOptions_Extrude;
  tr->ops->setup                 = DMPlexTransformSetUp_Extrude;
  tr->ops->destroy               = DMPlexTransformDestroy_Extrude;
  tr->ops->setdimensions         = DMPlexTransformSetDimensions_Extrude;
  tr->ops->celltransform         = DMPlexTransformCellTransform_Extrude;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_Extrude;
  tr->ops->mapcoordinates        = DMPlexTransformMapCoordinates_Extrude;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Extrude(DMPlexTransform tr)
{
  DMPlexTransform_Extrude *ex;
  DM                       dm;
  PetscInt                 dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&ex));
  tr->data      = ex;
  ex->thickness = 1.;
  ex->useTensor = PETSC_TRUE;
  ex->symmetric = PETSC_FALSE;
  ex->periodic  = PETSC_FALSE;
  ex->normalAlg = NORMAL_DEFAULT;
  ex->layerPos  = NULL;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &ex->cdim));
  PetscCall(DMPlexTransformExtrudeSetLayers(tr, 1));
  PetscCall(DMPlexTransformInitialize_Extrude(tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeGetLayers - Get the number of extruded layers.

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. layers - The number of layers

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeSetLayers()`
@*/
PetscErrorCode DMPlexTransformExtrudeGetLayers(DMPlexTransform tr, PetscInt *layers)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(layers, 2);
  *layers = ex->layers;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeSetLayers - Set the number of extruded layers.

  Not Collective

  Input Parameters:
+ tr     - The `DMPlexTransform`
- layers - The number of layers

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeGetLayers()`
@*/
PetscErrorCode DMPlexTransformExtrudeSetLayers(DMPlexTransform tr, PetscInt layers)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->layers = layers;
  PetscCall(PetscFree(ex->layerPos));
  PetscCall(PetscCalloc1(ex->layers + 1, &ex->layerPos));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeGetThickness - Get the total thickness of the layers

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. thickness - The total thickness of the layers

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeSetThickness()`
@*/
PetscErrorCode DMPlexTransformExtrudeGetThickness(DMPlexTransform tr, PetscReal *thickness)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(thickness, 2);
  *thickness = ex->thickness;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeSetThickness - Set the total thickness of the layers

  Not Collective

  Input Parameters:
+ tr        - The `DMPlexTransform`
- thickness - The total thickness of the layers

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeGetThickness()`
@*/
PetscErrorCode DMPlexTransformExtrudeSetThickness(DMPlexTransform tr, PetscReal thickness)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCheck(thickness > 0., PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_OUTOFRANGE, "Height of layers %g must be positive", (double)thickness);
  ex->thickness = thickness;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeGetTensor - Get the flag to use tensor cells

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. useTensor - The flag to use tensor cells

  Note:
  This flag determines the orientation behavior of the created points.

  For example, if tensor is `PETSC_TRUE`, then
.vb
  DM_POLYTOPE_POINT_PRISM_TENSOR is made instead of DM_POLYTOPE_SEGMENT,
  DM_POLYTOPE_SEG_PRISM_TENSOR instead of DM_POLYTOPE_QUADRILATERAL,
  DM_POLYTOPE_TRI_PRISM_TENSOR instead of DM_POLYTOPE_TRI_PRISM, and
  DM_POLYTOPE_QUAD_PRISM_TENSOR instead of DM_POLYTOPE_HEXAHEDRON.
.ve

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeSetTensor()`
@*/
PetscErrorCode DMPlexTransformExtrudeGetTensor(DMPlexTransform tr, PetscBool *useTensor)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(useTensor, 2);
  *useTensor = ex->useTensor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeSetTensor - Set the flag to use tensor cells

  Not Collective

  Input Parameters:
+ tr        - The `DMPlexTransform`
- useTensor - The flag for tensor cells

  Note:
  This flag determines the orientation behavior of the created points
  For example, if tensor is `PETSC_TRUE`, then
.vb
  DM_POLYTOPE_POINT_PRISM_TENSOR is made instead of DM_POLYTOPE_SEGMENT,
  DM_POLYTOPE_SEG_PRISM_TENSOR instead of DM_POLYTOPE_QUADRILATERAL,
  DM_POLYTOPE_TRI_PRISM_TENSOR instead of DM_POLYTOPE_TRI_PRISM, and
  DM_POLYTOPE_QUAD_PRISM_TENSOR instead of DM_POLYTOPE_HEXAHEDRON.
.ve

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeGetTensor()`
@*/
PetscErrorCode DMPlexTransformExtrudeSetTensor(DMPlexTransform tr, PetscBool useTensor)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->useTensor = useTensor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeGetSymmetric - Get the flag to extrude symmetrically from the initial surface

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. symmetric - The flag to extrude symmetrically

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeSetSymmetric()`, `DMPlexTransformExtrudeGetPeriodic()`
@*/
PetscErrorCode DMPlexTransformExtrudeGetSymmetric(DMPlexTransform tr, PetscBool *symmetric)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(symmetric, 2);
  *symmetric = ex->symmetric;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeSetSymmetric - Set the flag to extrude symmetrically from the initial surface

  Not Collective

  Input Parameters:
+ tr        - The `DMPlexTransform`
- symmetric - The flag to extrude symmetrically

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeGetSymmetric()`, `DMPlexTransformExtrudeSetPeriodic()`
@*/
PetscErrorCode DMPlexTransformExtrudeSetSymmetric(DMPlexTransform tr, PetscBool symmetric)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->symmetric = symmetric;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeGetPeriodic - Get the flag to extrude periodically from the initial surface

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. periodic - The flag to extrude periodically

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeSetPeriodic()`, `DMPlexTransformExtrudeGetSymmetric()`
@*/
PetscErrorCode DMPlexTransformExtrudeGetPeriodic(DMPlexTransform tr, PetscBool *periodic)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(periodic, 2);
  *periodic = ex->periodic;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeSetPeriodic - Set the flag to extrude periodically from the initial surface

  Not Collective

  Input Parameters:
+ tr       - The `DMPlexTransform`
- periodic - The flag to extrude periodically

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeGetPeriodic()`, `DMPlexTransformExtrudeSetSymmetric()`
@*/
PetscErrorCode DMPlexTransformExtrudeSetPeriodic(DMPlexTransform tr, PetscBool periodic)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->periodic = periodic;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeGetNormal - Get the extrusion normal vector

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. normal - The extrusion direction

  Note:
  The user passes in an array, which is filled by the library.

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeSetNormal()`
@*/
PetscErrorCode DMPlexTransformExtrudeGetNormal(DMPlexTransform tr, PetscReal normal[])
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;
  PetscInt                 d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  if (ex->normalAlg == NORMAL_INPUT) {
    for (d = 0; d < ex->cdimEx; ++d) normal[d] = ex->normal[d];
  } else {
    for (d = 0; d < ex->cdimEx; ++d) normal[d] = 0.;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeSetNormal - Set the extrusion normal

  Not Collective

  Input Parameters:
+ tr     - The `DMPlexTransform`
- normal - The extrusion direction

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeGetNormal()`
@*/
PetscErrorCode DMPlexTransformExtrudeSetNormal(DMPlexTransform tr, const PetscReal normal[])
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;
  PetscInt                 d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->normalAlg = NORMAL_INPUT;
  if (!ex->cdimEx) PetscCall(DMPlexTransformExtrudeComputeExtrusionDim(tr));
  for (d = 0; d < ex->cdimEx; ++d) ex->normal[d] = normal[d];
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@C
  DMPlexTransformExtrudeSetNormalFunction - Set a function to determine the extrusion normal

  Not Collective

  Input Parameters:
+ tr         - The `DMPlexTransform`
- normalFunc - A function determining the extrusion direction, see `PetscSimplePointFn` for the calling sequence

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeGetNormal()`, `PetscSimplePointFn`
@*/
PetscErrorCode DMPlexTransformExtrudeSetNormalFunction(DMPlexTransform tr, PetscSimplePointFn *normalFunc)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->normalFunc = normalFunc;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformExtrudeSetThicknesses - Set the thickness of each layer

  Not Collective

  Input Parameters:
+ tr          - The `DMPlexTransform`
. Nth         - The number of thicknesses
- thicknesses - The array of thicknesses

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformExtrudeSetThickness()`, `DMPlexTransformExtrudeGetThickness()`
@*/
PetscErrorCode DMPlexTransformExtrudeSetThicknesses(DMPlexTransform tr, PetscInt Nth, const PetscReal thicknesses[])
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *)tr->data;
  PetscInt                 t;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCheck(Nth > 0, PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_OUTOFRANGE, "Number of thicknesses %" PetscInt_FMT " must be positive", Nth);
  ex->Nth = PetscMin(Nth, ex->layers);
  PetscCall(PetscFree(ex->thicknesses));
  PetscCall(PetscMalloc1(ex->Nth, &ex->thicknesses));
  for (t = 0; t < ex->Nth; ++t) {
    PetscCheck(thicknesses[t] > 0., PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_OUTOFRANGE, "Thickness %g of layer %" PetscInt_FMT " must be positive", (double)thicknesses[t], t);
    ex->thicknesses[t] = thicknesses[t];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
