#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformView_Extrude(DMPlexTransform tr, PetscViewer viewer)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;
  PetscBool                isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject) tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Extrusion transformation %s\n", name ? name : ""));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  number of layers: %D\n", ex->layers));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  create tensor cells: %s\n", ex->useTensor ? "YES" : "NO"));
  } else {
    SETERRQ(PetscObjectComm((PetscObject) tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject) viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetFromOptions_Extrude(PetscOptionItems *PetscOptionsObject, DMPlexTransform tr)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;
  PetscReal                th, normal[3], *thicknesses;
  PetscInt                 nl, Nc;
  PetscBool                tensor, sym, flg;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 2);
  PetscCall(PetscOptionsHead(PetscOptionsObject, "DMPlexTransform Extrusion Options"));
  PetscCall(PetscOptionsBoundedInt("-dm_plex_transform_extrude_layers", "Number of layers to extrude", "", ex->layers, &nl, &flg, 1));
  if (flg) PetscCall(DMPlexTransformExtrudeSetLayers(tr, nl));
  PetscCall(PetscOptionsReal("-dm_plex_transform_extrude_thickness", "Total thickness of extruded layers", "", ex->thickness, &th, &flg));
  if (flg) PetscCall(DMPlexTransformExtrudeSetThickness(tr, th));
  PetscCall(PetscOptionsBool("-dm_plex_transform_extrude_use_tensor", "Create tensor cells", "", ex->useTensor, &tensor, &flg));
  if (flg) PetscCall(DMPlexTransformExtrudeSetTensor(tr, tensor));
  PetscCall(PetscOptionsBool("-dm_plex_transform_extrude_symmetric", "Extrude layers symmetrically about the surface", "", ex->symmetric, &sym, &flg));
  if (flg) PetscCall(DMPlexTransformExtrudeSetSymmetric(tr, sym));
  Nc = 3;
  PetscCall(PetscOptionsRealArray("-dm_plex_transform_extrude_normal", "Input normal vector for extrusion", "", normal, &Nc, &flg));
  if (flg) {
    PetscCheckFalse(Nc != ex->cdimEx,PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_SIZ, "Input normal has size %D != %D extruded coordinate dimension", Nc, ex->cdimEx);
    PetscCall(DMPlexTransformExtrudeSetNormal(tr, normal));
  }
  nl   = ex->layers;
  PetscCall(PetscMalloc1(nl, &thicknesses));
  PetscCall(PetscOptionsRealArray("-dm_plex_transform_extrude_thicknesses", "Thickness of each individual extruded layer", "", thicknesses, &nl, &flg));
  if (flg) {
    PetscCheck(nl,PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_OUTOFRANGE, "Must give at least one thickness for -dm_plex_transform_extrude_thicknesses");
    PetscCall(DMPlexTransformExtrudeSetThicknesses(tr, nl, thicknesses));
  }
  PetscCall(PetscFree(thicknesses));
  PetscCall(PetscOptionsTail());
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetDimensions_Extrude(DMPlexTransform tr, DM dm, DM tdm)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;
  PetscInt                 dim;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMSetDimension(tdm, dim+1));
  PetscCall(DMSetCoordinateDim(tdm, ex->cdimEx));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetUp_Extrude(DMPlexTransform tr)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;
  DM                       dm;
  DMPolytopeType           ct;
  PetscInt                 Nl = ex->layers, l, i, ict, Nc, No, coff, ooff;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(PetscMalloc5(DM_NUM_POLYTOPES, &ex->Nt, DM_NUM_POLYTOPES, &ex->target, DM_NUM_POLYTOPES, &ex->size, DM_NUM_POLYTOPES, &ex->cone, DM_NUM_POLYTOPES, &ex->ornt));
  for (ict = 0; ict < DM_NUM_POLYTOPES; ++ict) {
    ex->Nt[ict]     = -1;
    ex->target[ict] = NULL;
    ex->size[ict]   = NULL;
    ex->cone[ict]   = NULL;
    ex->ornt[ict]   = NULL;
  }
  /* DM_POLYTOPE_POINT produces Nl+1 points and Nl segments/tensor segments */
  ct = DM_POLYTOPE_POINT;
  ex->Nt[ct] = 2;
  Nc = 6*Nl;
  No = 2*Nl;
  PetscCall(PetscMalloc4(ex->Nt[ct], &ex->target[ct], ex->Nt[ct], &ex->size[ct], Nc, &ex->cone[ct], No, &ex->ornt[ct]));
  ex->target[ct][0] = DM_POLYTOPE_POINT;
  ex->target[ct][1] = ex->useTensor ? DM_POLYTOPE_POINT_PRISM_TENSOR : DM_POLYTOPE_SEGMENT;
  ex->size[ct][0]   = Nl+1;
  ex->size[ct][1]   = Nl;
  /*   cones for segments/tensor segments */
  for (i = 0; i < Nl; ++i) {
    ex->cone[ct][6*i+0] = DM_POLYTOPE_POINT;
    ex->cone[ct][6*i+1] = 0;
    ex->cone[ct][6*i+2] = i;
    ex->cone[ct][6*i+3] = DM_POLYTOPE_POINT;
    ex->cone[ct][6*i+4] = 0;
    ex->cone[ct][6*i+5] = i+1;
  }
  for (i = 0; i < No; ++i) ex->ornt[ct][i] = 0;
  /* DM_POLYTOPE_SEGMENT produces Nl+1 segments and Nl quads/tensor quads */
  ct = DM_POLYTOPE_SEGMENT;
  ex->Nt[ct] = 2;
  Nc = 8*(Nl+1) + 14*Nl;
  No = 2*(Nl+1) + 4*Nl;
  PetscCall(PetscMalloc4(ex->Nt[ct], &ex->target[ct], ex->Nt[ct], &ex->size[ct], Nc, &ex->cone[ct], No, &ex->ornt[ct]));
  ex->target[ct][0] = DM_POLYTOPE_SEGMENT;
  ex->target[ct][1] = ex->useTensor ? DM_POLYTOPE_SEG_PRISM_TENSOR : DM_POLYTOPE_QUADRILATERAL;
  ex->size[ct][0]   = Nl+1;
  ex->size[ct][1]   = Nl;
  /*   cones for segments */
  for (i = 0; i < Nl+1; ++i) {
    ex->cone[ct][8*i+0] = DM_POLYTOPE_POINT;
    ex->cone[ct][8*i+1] = 1;
    ex->cone[ct][8*i+2] = 0;
    ex->cone[ct][8*i+3] = i;
    ex->cone[ct][8*i+4] = DM_POLYTOPE_POINT;
    ex->cone[ct][8*i+5] = 1;
    ex->cone[ct][8*i+6] = 1;
    ex->cone[ct][8*i+7] = i;
  }
  for (i = 0; i < 2*(Nl+1); ++i) ex->ornt[ct][i] = 0;
  /*   cones for quads/tensor quads */
  coff = 8*(Nl+1);
  ooff = 2*(Nl+1);
  for (i = 0; i < Nl; ++i) {
    if (ex->useTensor) {
      ex->cone[ct][coff+14*i+0]  = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff+14*i+1]  = 0;
      ex->cone[ct][coff+14*i+2]  = i;
      ex->cone[ct][coff+14*i+3]  = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff+14*i+4]  = 0;
      ex->cone[ct][coff+14*i+5]  = i+1;
      ex->cone[ct][coff+14*i+6]  = DM_POLYTOPE_POINT_PRISM_TENSOR;
      ex->cone[ct][coff+14*i+7]  = 1;
      ex->cone[ct][coff+14*i+8]  = 0;
      ex->cone[ct][coff+14*i+9]  = i;
      ex->cone[ct][coff+14*i+10] = DM_POLYTOPE_POINT_PRISM_TENSOR;
      ex->cone[ct][coff+14*i+11] = 1;
      ex->cone[ct][coff+14*i+12] = 1;
      ex->cone[ct][coff+14*i+13] = i;
      ex->ornt[ct][ooff+4*i+0] = 0;
      ex->ornt[ct][ooff+4*i+1] = 0;
      ex->ornt[ct][ooff+4*i+2] = 0;
      ex->ornt[ct][ooff+4*i+3] = 0;
    } else {
      ex->cone[ct][coff+14*i+0]  = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff+14*i+1]  = 0;
      ex->cone[ct][coff+14*i+2]  = i;
      ex->cone[ct][coff+14*i+3]  = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff+14*i+4]  = 1;
      ex->cone[ct][coff+14*i+5]  = 1;
      ex->cone[ct][coff+14*i+6]  = i;
      ex->cone[ct][coff+14*i+7]  = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff+14*i+8]  = 0;
      ex->cone[ct][coff+14*i+9]  = i+1;
      ex->cone[ct][coff+14*i+10] = DM_POLYTOPE_SEGMENT;
      ex->cone[ct][coff+14*i+11] = 1;
      ex->cone[ct][coff+14*i+12] = 0;
      ex->cone[ct][coff+14*i+13] = i;
      ex->ornt[ct][ooff+4*i+0] =  0;
      ex->ornt[ct][ooff+4*i+1] =  0;
      ex->ornt[ct][ooff+4*i+2] = -1;
      ex->ornt[ct][ooff+4*i+3] = -1;
    }
  }
  /* DM_POLYTOPE_TRIANGLE produces Nl+1 triangles and Nl triangular prisms/tensor triangular prisms */
  ct = DM_POLYTOPE_TRIANGLE;
  ex->Nt[ct] = 2;
  Nc = 12*(Nl+1) + 18*Nl;
  No =  3*(Nl+1) +  5*Nl;
  PetscCall(PetscMalloc4(ex->Nt[ct], &ex->target[ct], ex->Nt[ct], &ex->size[ct], Nc, &ex->cone[ct], No, &ex->ornt[ct]));
  ex->target[ct][0] = DM_POLYTOPE_TRIANGLE;
  ex->target[ct][1] = ex->useTensor ? DM_POLYTOPE_TRI_PRISM_TENSOR : DM_POLYTOPE_TRI_PRISM;
  ex->size[ct][0]   = Nl+1;
  ex->size[ct][1]   = Nl;
  /*   cones for triangles */
  for (i = 0; i < Nl+1; ++i) {
    ex->cone[ct][12*i+0]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][12*i+1]  = 1;
    ex->cone[ct][12*i+2]  = 0;
    ex->cone[ct][12*i+3]  = i;
    ex->cone[ct][12*i+4]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][12*i+5]  = 1;
    ex->cone[ct][12*i+6]  = 1;
    ex->cone[ct][12*i+7]  = i;
    ex->cone[ct][12*i+8]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][12*i+9]  = 1;
    ex->cone[ct][12*i+10] = 2;
    ex->cone[ct][12*i+11] = i;
  }
  for (i = 0; i < 3*(Nl+1); ++i) ex->ornt[ct][i] = 0;
  /*   cones for triangular prisms/tensor triangular prisms */
  coff = 12*(Nl+1);
  ooff = 3*(Nl+1);
  for (i = 0; i < Nl; ++i) {
    if (ex->useTensor) {
      ex->cone[ct][coff+18*i+0]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[ct][coff+18*i+1]  = 0;
      ex->cone[ct][coff+18*i+2]  = i;
      ex->cone[ct][coff+18*i+3]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[ct][coff+18*i+4]  = 0;
      ex->cone[ct][coff+18*i+5]  = i+1;
      ex->cone[ct][coff+18*i+6]  = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff+18*i+7]  = 1;
      ex->cone[ct][coff+18*i+8]  = 0;
      ex->cone[ct][coff+18*i+9]  = i;
      ex->cone[ct][coff+18*i+10] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff+18*i+11] = 1;
      ex->cone[ct][coff+18*i+12] = 1;
      ex->cone[ct][coff+18*i+13] = i;
      ex->cone[ct][coff+18*i+14] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff+18*i+15] = 1;
      ex->cone[ct][coff+18*i+16] = 2;
      ex->cone[ct][coff+18*i+17] = i;
      ex->ornt[ct][ooff+5*i+0] = 0;
      ex->ornt[ct][ooff+5*i+1] = 0;
      ex->ornt[ct][ooff+5*i+2] = 0;
      ex->ornt[ct][ooff+5*i+3] = 0;
      ex->ornt[ct][ooff+5*i+4] = 0;
    } else {
      ex->cone[ct][coff+18*i+0]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[ct][coff+18*i+1]  = 0;
      ex->cone[ct][coff+18*i+2]  = i;
      ex->cone[ct][coff+18*i+3]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[ct][coff+18*i+4]  = 0;
      ex->cone[ct][coff+18*i+5]  = i+1;
      ex->cone[ct][coff+18*i+6]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+18*i+7]  = 1;
      ex->cone[ct][coff+18*i+8]  = 0;
      ex->cone[ct][coff+18*i+9]  = i;
      ex->cone[ct][coff+18*i+10] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+18*i+11] = 1;
      ex->cone[ct][coff+18*i+12] = 1;
      ex->cone[ct][coff+18*i+13] = i;
      ex->cone[ct][coff+18*i+14] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+18*i+15] = 1;
      ex->cone[ct][coff+18*i+16] = 2;
      ex->cone[ct][coff+18*i+17] = i;
      ex->ornt[ct][ooff+5*i+0] = -2;
      ex->ornt[ct][ooff+5*i+1] =  0;
      ex->ornt[ct][ooff+5*i+2] =  0;
      ex->ornt[ct][ooff+5*i+3] =  0;
      ex->ornt[ct][ooff+5*i+4] =  0;
    }
  }
  /* DM_POLYTOPE_QUADRILATERAL produces Nl+1 quads and Nl hexes/tensor hexes */
  ct = DM_POLYTOPE_QUADRILATERAL;
  ex->Nt[ct] = 2;
  Nc = 16*(Nl+1) + 22*Nl;
  No =  4*(Nl+1) +  6*Nl;
  PetscCall(PetscMalloc4(ex->Nt[ct], &ex->target[ct], ex->Nt[ct], &ex->size[ct], Nc, &ex->cone[ct], No, &ex->ornt[ct]));
  ex->target[ct][0] = DM_POLYTOPE_QUADRILATERAL;
  ex->target[ct][1] = ex->useTensor ? DM_POLYTOPE_QUAD_PRISM_TENSOR : DM_POLYTOPE_HEXAHEDRON;
  ex->size[ct][0]   = Nl+1;
  ex->size[ct][1]   = Nl;
  /*   cones for quads */
  for (i = 0; i < Nl+1; ++i) {
    ex->cone[ct][16*i+0]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][16*i+1]  = 1;
    ex->cone[ct][16*i+2]  = 0;
    ex->cone[ct][16*i+3]  = i;
    ex->cone[ct][16*i+4]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][16*i+5]  = 1;
    ex->cone[ct][16*i+6]  = 1;
    ex->cone[ct][16*i+7]  = i;
    ex->cone[ct][16*i+8]  = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][16*i+9]  = 1;
    ex->cone[ct][16*i+10] = 2;
    ex->cone[ct][16*i+11] = i;
    ex->cone[ct][16*i+12] = DM_POLYTOPE_SEGMENT;
    ex->cone[ct][16*i+13] = 1;
    ex->cone[ct][16*i+14] = 3;
    ex->cone[ct][16*i+15] = i;
  }
  for (i = 0; i < 4*(Nl+1); ++i) ex->ornt[ct][i] = 0;
  /*   cones for hexes/tensor hexes */
  coff = 16*(Nl+1);
  ooff = 4*(Nl+1);
  for (i = 0; i < Nl; ++i) {
    if (ex->useTensor) {
      ex->cone[ct][coff+22*i+0]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+22*i+1]  = 0;
      ex->cone[ct][coff+22*i+2]  = i;
      ex->cone[ct][coff+22*i+3]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+22*i+4]  = 0;
      ex->cone[ct][coff+22*i+5]  = i+1;
      ex->cone[ct][coff+22*i+6]  = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff+22*i+7]  = 1;
      ex->cone[ct][coff+22*i+8]  = 0;
      ex->cone[ct][coff+22*i+9]  = i;
      ex->cone[ct][coff+22*i+10] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff+22*i+11] = 1;
      ex->cone[ct][coff+22*i+12] = 1;
      ex->cone[ct][coff+22*i+13] = i;
      ex->cone[ct][coff+22*i+14] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff+22*i+15] = 1;
      ex->cone[ct][coff+22*i+16] = 2;
      ex->cone[ct][coff+22*i+17] = i;
      ex->cone[ct][coff+22*i+18] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[ct][coff+22*i+19] = 1;
      ex->cone[ct][coff+22*i+20] = 3;
      ex->cone[ct][coff+22*i+21] = i;
      ex->ornt[ct][ooff+6*i+0] = 0;
      ex->ornt[ct][ooff+6*i+1] = 0;
      ex->ornt[ct][ooff+6*i+2] = 0;
      ex->ornt[ct][ooff+6*i+3] = 0;
      ex->ornt[ct][ooff+6*i+4] = 0;
      ex->ornt[ct][ooff+6*i+5] = 0;
    } else {
      ex->cone[ct][coff+22*i+0]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+22*i+1]  = 0;
      ex->cone[ct][coff+22*i+2]  = i;
      ex->cone[ct][coff+22*i+3]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+22*i+4]  = 0;
      ex->cone[ct][coff+22*i+5]  = i+1;
      ex->cone[ct][coff+22*i+6]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+22*i+7]  = 1;
      ex->cone[ct][coff+22*i+8]  = 0;
      ex->cone[ct][coff+22*i+9]  = i;
      ex->cone[ct][coff+22*i+10] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+22*i+11] = 1;
      ex->cone[ct][coff+22*i+12] = 2;
      ex->cone[ct][coff+22*i+13] = i;
      ex->cone[ct][coff+22*i+14] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+22*i+15] = 1;
      ex->cone[ct][coff+22*i+16] = 1;
      ex->cone[ct][coff+22*i+17] = i;
      ex->cone[ct][coff+22*i+18] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[ct][coff+22*i+19] = 1;
      ex->cone[ct][coff+22*i+20] = 3;
      ex->cone[ct][coff+22*i+21] = i;
      ex->ornt[ct][ooff+6*i+0] = -2;
      ex->ornt[ct][ooff+6*i+1] =  0;
      ex->ornt[ct][ooff+6*i+2] =  0;
      ex->ornt[ct][ooff+6*i+3] =  0;
      ex->ornt[ct][ooff+6*i+4] =  0;
      ex->ornt[ct][ooff+6*i+5] =  1;
    }
  }
  /* Layers positions */
  if (!ex->Nth) {
    if (ex->symmetric) for (l = 0; l <= ex->layers; ++l) ex->layerPos[l] = (ex->thickness*l)/ex->layers - ex->thickness/2;
    else               for (l = 0; l <= ex->layers; ++l) ex->layerPos[l] = (ex->thickness*l)/ex->layers;
  } else {
    ex->thickness   = 0.;
    ex->layerPos[0] = 0.;
    for (l = 0; l < ex->layers; ++l) {
      const PetscReal t = ex->thicknesses[PetscMin(l, ex->Nth-1)];
      ex->layerPos[l+1] = ex->layerPos[l] + t;
      ex->thickness    += t;
    }
    if (ex->symmetric) for (l = 0; l <= ex->layers; ++l) ex->layerPos[l] -= ex->thickness/2.;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_Extrude(DMPlexTransform tr)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;
  PetscInt                 ct;

  PetscFunctionBegin;
  for (ct = 0; ct < DM_NUM_POLYTOPES; ++ct) {
    PetscCall(PetscFree4(ex->target[ct], ex->size[ct], ex->cone[ct], ex->ornt[ct]));
  }
  PetscCall(PetscFree5(ex->Nt, ex->target, ex->size, ex->cone, ex->ornt));
  PetscCall(PetscFree(ex->layerPos));
  PetscCall(PetscFree(ex));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_Extrude(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;

  PetscFunctionBeginHot;
  *rnew = r;
  *onew = DMPolytopeTypeComposeOrientation(tct, o, so);
  if (!so) PetscFunctionReturn(0);
  if (ex->useTensor) {
    switch (sct) {
      case DM_POLYTOPE_POINT: break;
      case DM_POLYTOPE_SEGMENT:
      switch (tct) {
        case DM_POLYTOPE_SEGMENT: break;
        case DM_POLYTOPE_SEG_PRISM_TENSOR:
          *onew = DMPolytopeTypeComposeOrientation(tct, o, so ? -1 : 0);
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell type %s", DMPolytopeTypes[sct]);
    }
  } else {
    switch (sct) {
      case DM_POLYTOPE_POINT: break;
      case DM_POLYTOPE_SEGMENT:
      switch (tct) {
        case DM_POLYTOPE_SEGMENT: break;
        case DM_POLYTOPE_QUADRILATERAL:
          *onew = DMPolytopeTypeComposeOrientation(tct, o, so ? -3 : 0);
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell type %s", DMPolytopeTypes[sct]);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCellTransform_Extrude(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;

  PetscFunctionBegin;
  if (rt) *rt = 0;
  if (ex->Nt[source] < 0) {
    /* Ignore cells that cannot be extruded */
    *Nt     = 0;
    *target = NULL;
    *size   = NULL;
    *cone   = NULL;
    *ornt   = NULL;
  } else {
    *Nt     = ex->Nt[source];
    *target = ex->target[source];
    *size   = ex->size[source];
    *cone   = ex->cone[source];
    *ornt   = ex->ornt[source];
  }
  PetscFunctionReturn(0);
}

/* Computes new vertex along normal */
static PetscErrorCode DMPlexTransformMapCoordinates_Extrude(DMPlexTransform tr, DMPolytopeType pct, DMPolytopeType ct, PetscInt p, PetscInt r, PetscInt Nv, PetscInt dE, const PetscScalar in[], PetscScalar out[])
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;
  DM                       dm;
  PetscReal                ones2[2]  = {0., 1.}, ones3[3] = { 0., 0., 1.};
  PetscReal                normal[3] = {0., 0., 0.}, norm;
  PetscBool                computeNormal;
  PetscInt                 dim, dEx = ex->cdimEx, cStart, cEnd, d;

  PetscFunctionBeginHot;
  PetscCheckFalse(pct != DM_POLYTOPE_POINT,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for parent point type %s",DMPolytopeTypes[pct]);
  PetscCheckFalse(ct  != DM_POLYTOPE_POINT,PETSC_COMM_SELF,PETSC_ERR_SUP,"Not for refined point type %s",DMPolytopeTypes[ct]);
  PetscCheckFalse(Nv != 1,PETSC_COMM_SELF,PETSC_ERR_SUP,"Vertices should be produced from a single vertex, not %D",Nv);
  PetscCheckFalse(dE != ex->cdim,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Coordinate dim %D != %D original dimension", dE, ex->cdim);

  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  computeNormal = dim != ex->cdim && !ex->useNormal ? PETSC_TRUE : PETSC_FALSE;
  if (computeNormal) {
    PetscInt *closure = NULL;
    PetscInt  closureSize, cl;

    PetscCall(DMPlexGetSimplexOrBoxCells(dm, 0, &cStart, &cEnd));
    PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_FALSE, &closureSize, &closure));
    for (cl = 0; cl < closureSize*2; cl += 2) {
      if ((closure[cl] >= cStart) && (closure[cl] < cEnd)) {
        PetscReal cnormal[3] = {0, 0, 0};

        PetscCall(DMPlexComputeCellGeometryFVM(dm, closure[cl], NULL, NULL, cnormal));
        for (d = 0; d < dEx; ++d) normal[d] += cnormal[d];
      }
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_FALSE, &closureSize, &closure));
  } else if (ex->useNormal) {
    for (d = 0; d < dEx; ++d) normal[d] = ex->normal[d];
  } else if (ex->cdimEx == 2) {
    for (d = 0; d < dEx; ++d) normal[d] = ones2[d];
  } else if (ex->cdimEx == 3) {
    for (d = 0; d < dEx; ++d) normal[d] = ones3[d];
  } else SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to determine normal for extrusion");

  for (d = 0, norm = 0.0; d < dEx; ++d) norm += PetscSqr(normal[d]);
  for (d = 0; d < dEx; ++d) normal[d] *= 1./PetscSqrtReal(norm);
  for (d = 0; d < dEx;      ++d) out[d]  = normal[d]*ex->layerPos[r];
  for (d = 0; d < ex->cdim; ++d) out[d] += in[d];
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformInitialize_Extrude(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view           = DMPlexTransformView_Extrude;
  tr->ops->setfromoptions = DMPlexTransformSetFromOptions_Extrude;
  tr->ops->setup          = DMPlexTransformSetUp_Extrude;
  tr->ops->destroy        = DMPlexTransformDestroy_Extrude;
  tr->ops->setdimensions  = DMPlexTransformSetDimensions_Extrude;
  tr->ops->celltransform  = DMPlexTransformCellTransform_Extrude;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_Extrude;
  tr->ops->mapcoordinates = DMPlexTransformMapCoordinates_Extrude;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Extrude(DMPlexTransform tr)
{
  DMPlexTransform_Extrude *ex;
  DM                       dm;
  PetscInt                 dim;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNewLog(tr, &ex));
  tr->data        = ex;
  ex->thickness   = 1.;
  ex->useTensor   = PETSC_TRUE;
  ex->symmetric   = PETSC_FALSE;
  ex->useNormal   = PETSC_FALSE;
  ex->layerPos    = NULL;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMGetCoordinateDim(dm, &ex->cdim));
  ex->cdimEx = ex->cdim == dim ? ex->cdim+1 : ex->cdim;
  PetscCall(DMPlexTransformExtrudeSetLayers(tr, 1));
  PetscCall(DMPlexTransformInitialize_Extrude(tr));
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformExtrudeGetLayers - Get the number of extruded layers.

  Not collective

  Input Parameter:
. tr  - The DMPlexTransform

  Output Parameter:
. layers - The number of layers

  Level: intermediate

.seealso: DMPlexTransformExtrudeSetLayers()
@*/
PetscErrorCode DMPlexTransformExtrudeGetLayers(DMPlexTransform tr, PetscInt *layers)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidIntPointer(layers, 2);
  *layers = ex->layers;
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformExtrudeSetLayers - Set the number of extruded layers.

  Not collective

  Input Parameters:
+ tr  - The DMPlexTransform
- layers - The number of layers

  Level: intermediate

.seealso: DMPlexTransformExtrudeGetLayers()
@*/
PetscErrorCode DMPlexTransformExtrudeSetLayers(DMPlexTransform tr, PetscInt layers)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->layers = layers;
  PetscCall(PetscFree(ex->layerPos));
  PetscCall(PetscCalloc1(ex->layers+1, &ex->layerPos));
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformExtrudeGetThickness - Get the total thickness of the layers

  Not collective

  Input Parameter:
. tr  - The DMPlexTransform

  Output Parameter:
. thickness - The total thickness of the layers

  Level: intermediate

.seealso: DMPlexTransformExtrudeSetThickness()
@*/
PetscErrorCode DMPlexTransformExtrudeGetThickness(DMPlexTransform tr, PetscReal *thickness)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidRealPointer(thickness, 2);
  *thickness = ex->thickness;
  PetscFunctionReturn(0);
}

/*@
  DMPlexTransformExtrudeSetThickness - Set the total thickness of the layers

  Not collective

  Input Parameters:
+ tr  - The DMPlexTransform
- thickness - The total thickness of the layers

  Level: intermediate

.seealso: DMPlexTransformExtrudeGetThickness()
@*/
PetscErrorCode DMPlexTransformExtrudeSetThickness(DMPlexTransform tr, PetscReal thickness)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCheckFalse(thickness <= 0.,PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_OUTOFRANGE, "Height of layers %g must be positive", (double) thickness);
  ex->thickness = thickness;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformExtrudeGetTensor(DMPlexTransform tr, PetscBool *useTensor)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidBoolPointer(useTensor, 2);
  *useTensor = ex->useTensor;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformExtrudeSetTensor(DMPlexTransform tr, PetscBool useTensor)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->useTensor = useTensor;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformExtrudeGetSymmetric(DMPlexTransform tr, PetscBool *symmetric)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidBoolPointer(symmetric, 2);
  *symmetric = ex->symmetric;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformExtrudeSetSymmetric(DMPlexTransform tr, PetscBool symmetric)
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->symmetric = symmetric;
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformExtrudeSetNormal(DMPlexTransform tr, const PetscReal normal[])
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;
  PetscInt                 d;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->useNormal = PETSC_TRUE;
  for (d = 0; d < ex->cdimEx; ++d) ex->normal[d] = normal[d];
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformExtrudeSetThicknesses(DMPlexTransform tr, PetscInt Nth, const PetscReal thicknesses[])
{
  DMPlexTransform_Extrude *ex = (DMPlexTransform_Extrude *) tr->data;
  PetscInt                 t;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCheckFalse(Nth <= 0,PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_OUTOFRANGE, "Number of thicknesses %D must be positive", Nth);
  ex->Nth = PetscMin(Nth, ex->layers);
  PetscCall(PetscFree(ex->thicknesses));
  PetscCall(PetscMalloc1(ex->Nth, &ex->thicknesses));
  for (t = 0; t < ex->Nth; ++t) {
    PetscCheckFalse(thicknesses[t] <= 0.,PetscObjectComm((PetscObject) tr), PETSC_ERR_ARG_OUTOFRANGE, "Thickness %g of layer %D must be positive", (double) thicknesses[t], t);
    ex->thicknesses[t] = thicknesses[t];
  }
  PetscFunctionReturn(0);
}
