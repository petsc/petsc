#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

#include <petsc/private/dmlabelimpl.h> // For DMLabelMakeAllInvalid_Internal()

/*
  The cohesive transformation extrudes cells into a mesh from faces along an internal boundary.

  Orientation:

  We will say that a face has a positive and negative side. The positive side is defined by the cell which attaches the face with a positive orientation, and the negative side cell attaches it with a negative orientation (a reflection). However, this means that the positive side is in the opposite direction of the face normal, and the negative side is in the direction of the face normal, since all cells have outward facing normals. For clarity, in 2D the cross product of the normal and the edge is in the positive z direction.

  Labeling:

  We require an active label on input, which marks all points on the internal surface. Each point is
  labeled with its depth. This label is passed to DMPlexLabelCohesiveComplete(), which adds all points
  which ``impinge'' on the surface, meaning a point has a face on the surface. These points are labeled
  with celltype + 100 on the positive side, and -(celltype + 100) on the negative side.

  Point Creation:

  We split points on the fault surface, creating a new partner point for each one. The negative side
  receives the old point, while the positive side receives the new partner. In addition, points are
  created with the two split points as boundaries. For example, split vertices have a segment between
  them, split edges a quadrilaterial, split triangles a prism, and split quads a hexahedron. By
  default, these spanning points have tensor ordering, but the user can choose to have them use the
  outward normal convention instead.

*/

static PetscErrorCode DMPlexTransformView_Cohesive(DMPlexTransform tr, PetscViewer viewer)
{
  DMPlexTransform_Cohesive *ex = (DMPlexTransform_Cohesive *)tr->data;
  PetscBool                 isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Cohesive extrusion transformation %s\n", name ? name : ""));
    PetscCall(PetscViewerASCIIPrintf(viewer, "  create tensor cells: %s\n", ex->useTensor ? "YES" : "NO"));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetFromOptions_Cohesive(DMPlexTransform tr, PetscOptionItems PetscOptionsObject)
{
  DMPlexTransform_Cohesive *ex = (DMPlexTransform_Cohesive *)tr->data;
  PetscReal                 width;
  PetscBool                 tensor, flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMPlexTransform Cohesive Extrusion Options");
  PetscCall(PetscOptionsBool("-dm_plex_transform_extrude_use_tensor", "Create tensor cells", "", ex->useTensor, &tensor, &flg));
  if (flg) PetscCall(DMPlexTransformCohesiveExtrudeSetTensor(tr, tensor));
  PetscCall(PetscOptionsReal("-dm_plex_transform_cohesive_width", "Width of a cohesive cell", "", ex->width, &width, &flg));
  if (flg) PetscCall(DMPlexTransformCohesiveExtrudeSetWidth(tr, width));
  PetscCall(PetscOptionsInt("-dm_plex_transform_cohesive_debug", "Det debugging level", "", ex->debug, &ex->debug, NULL));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeSplitFaceNumber - Compute an encoding describing which faces of p are split by the surface

  Not collective

  Input Parameters:
  + dm    - The `DM`
  . label - `DMLabel` marking the surface and adjacent points
  - p     - Impinging point, adjacent to the surface

  Output Parameter:
  . fsplit - A number encoding the faces which are split by the surface

  Level: developer

  Note: We will use a bit encoding, where bit k is 1 if face k is split.

.seealso: ComputeUnsplitFaceNumber()
*/
static PetscErrorCode ComputeSplitFaceNumber(DM dm, DMLabel label, PetscInt p, PetscInt *fsplit)
{
  const PetscInt *cone;
  PetscInt        coneSize, val;

  PetscFunctionBegin;
  *fsplit = 0;
  PetscCall(DMPlexGetCone(dm, p, &cone));
  PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
  PetscCheck(coneSize < (PetscInt)sizeof(*fsplit) * 8, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cone of size %" PetscInt_FMT " is too large to be contained in an integer", coneSize);
  for (PetscInt c = 0; c < coneSize; ++c) {
    PetscCall(DMLabelGetValue(label, cone[c], &val));
    if (val >= 0 && val < 100) *fsplit |= 1 << c;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  ComputeUnsplitFaceNumber - Compute an encoding describing which faces of p are unsplit by the surface

  Not collective

  Input Parameters:
  + dm    - The `DM`
  . label - `DMLabel` marking the surface and adjacent points
  - p     - Split point, on the surface

  Output Parameter:
  . funsplit - A number encoding the faces which are split by the surface

  Level: developer

  Note: We will use a bit encoding, where bit k is 1 if face k is unsplit.

.seealso: ComputeSplitFaceNumber()
*/
static PetscErrorCode ComputeUnsplitFaceNumber(DM dm, DMLabel label, PetscInt p, PetscInt *funsplit)
{
  const PetscInt *cone;
  PetscInt        coneSize, val;

  PetscFunctionBegin;
  *funsplit = 0;
  PetscCall(DMPlexGetCone(dm, p, &cone));
  PetscCall(DMPlexGetConeSize(dm, p, &coneSize));
  PetscCheck(coneSize < (PetscInt)sizeof(*funsplit) * 8, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cone of size %" PetscInt_FMT " is too large to be contained in an integer", coneSize);
  for (PetscInt c = 0; c < coneSize; ++c) {
    PetscCall(DMLabelGetValue(label, cone[c], &val));
    if (val >= 200) *funsplit |= 1 << c;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DM_POLYTOPE_POINT produces
     2 points when split, or 1 point when unsplit, and
     1 segment, or tensor segment
*/
static PetscErrorCode DMPlexTransformCohesiveExtrudeSetUp_Point(DMPlexTransform_Cohesive *ex)
{
  PetscInt rt, Nc, No;

  PetscFunctionBegin;
  // Unsplit vertex
  rt         = DM_POLYTOPE_POINT * 2 + 1;
  ex->Nt[rt] = 2;
  Nc         = 6;
  No         = 2;
  PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
  ex->target[rt][0] = DM_POLYTOPE_POINT;
  ex->target[rt][1] = ex->useTensor ? DM_POLYTOPE_POINT_PRISM_TENSOR : DM_POLYTOPE_SEGMENT;
  ex->size[rt][0]   = 1;
  ex->size[rt][1]   = 1;
  //   cone for segment/tensor segment
  ex->cone[rt][0] = DM_POLYTOPE_POINT;
  ex->cone[rt][1] = 0;
  ex->cone[rt][2] = 0;
  ex->cone[rt][3] = DM_POLYTOPE_POINT;
  ex->cone[rt][4] = 0;
  ex->cone[rt][5] = 0;
  for (PetscInt i = 0; i < No; ++i) ex->ornt[rt][i] = 0;
  // Split vertex
  rt         = (DM_POLYTOPE_POINT * 2 + 1) * 100 + 0;
  ex->Nt[rt] = 2;
  Nc         = 6;
  No         = 2;
  PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
  ex->target[rt][0] = DM_POLYTOPE_POINT;
  ex->target[rt][1] = ex->useTensor ? DM_POLYTOPE_POINT_PRISM_TENSOR : DM_POLYTOPE_SEGMENT;
  ex->size[rt][0]   = 2;
  ex->size[rt][1]   = 1;
  //   cone for segment/tensor segment
  ex->cone[rt][0] = DM_POLYTOPE_POINT;
  ex->cone[rt][1] = 0;
  ex->cone[rt][2] = 0;
  ex->cone[rt][3] = DM_POLYTOPE_POINT;
  ex->cone[rt][4] = 0;
  ex->cone[rt][5] = 1;
  for (PetscInt i = 0; i < No; ++i) ex->ornt[rt][i] = 0;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DM_POLYTOPE_SEGMENT produces
     2 segments when split, or 1 segment when unsplit, and
     1 quad, or tensor quad
*/
static PetscErrorCode DMPlexTransformCohesiveExtrudeSetUp_Segment(DMPlexTransform_Cohesive *ex)
{
  PetscInt rt, Nc, No, coff, ooff;

  PetscFunctionBegin;
  // Unsplit segment
  rt         = DM_POLYTOPE_SEGMENT * 2 + 1;
  ex->Nt[rt] = 2;
  Nc         = 8 + 14;
  No         = 2 + 4;
  PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
  ex->target[rt][0] = DM_POLYTOPE_SEGMENT;
  ex->target[rt][1] = ex->useTensor ? DM_POLYTOPE_SEG_PRISM_TENSOR : DM_POLYTOPE_QUADRILATERAL;
  ex->size[rt][0]   = 1;
  ex->size[rt][1]   = 1;
  //   cones for segment
  ex->cone[rt][0] = DM_POLYTOPE_POINT;
  ex->cone[rt][1] = 1;
  ex->cone[rt][2] = 0;
  ex->cone[rt][3] = 0;
  ex->cone[rt][4] = DM_POLYTOPE_POINT;
  ex->cone[rt][5] = 1;
  ex->cone[rt][6] = 1;
  ex->cone[rt][7] = 0;
  for (PetscInt i = 0; i < 2; ++i) ex->ornt[rt][i] = 0;
  //   cone for quad/tensor quad
  coff = 8;
  ooff = 2;
  if (ex->useTensor) {
    ex->cone[rt][coff + 0]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][coff + 1]  = 0;
    ex->cone[rt][coff + 2]  = 0;
    ex->cone[rt][coff + 3]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][coff + 4]  = 0;
    ex->cone[rt][coff + 5]  = 0;
    ex->cone[rt][coff + 6]  = DM_POLYTOPE_POINT_PRISM_TENSOR;
    ex->cone[rt][coff + 7]  = 1;
    ex->cone[rt][coff + 8]  = 0;
    ex->cone[rt][coff + 9]  = 0;
    ex->cone[rt][coff + 10] = DM_POLYTOPE_POINT_PRISM_TENSOR;
    ex->cone[rt][coff + 11] = 1;
    ex->cone[rt][coff + 12] = 1;
    ex->cone[rt][coff + 13] = 0;
    ex->ornt[rt][ooff + 0]  = 0;
    ex->ornt[rt][ooff + 1]  = 0;
    ex->ornt[rt][ooff + 2]  = 0;
    ex->ornt[rt][ooff + 3]  = 0;
  } else {
    ex->cone[rt][coff + 0]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][coff + 1]  = 0;
    ex->cone[rt][coff + 2]  = 0;
    ex->cone[rt][coff + 3]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][coff + 4]  = 1;
    ex->cone[rt][coff + 5]  = 1;
    ex->cone[rt][coff + 6]  = 0;
    ex->cone[rt][coff + 7]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][coff + 8]  = 0;
    ex->cone[rt][coff + 9]  = 0;
    ex->cone[rt][coff + 10] = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][coff + 11] = 1;
    ex->cone[rt][coff + 12] = 0;
    ex->cone[rt][coff + 13] = 0;
    ex->ornt[rt][ooff + 0]  = 0;
    ex->ornt[rt][ooff + 1]  = 0;
    ex->ornt[rt][ooff + 2]  = -1;
    ex->ornt[rt][ooff + 3]  = -1;
  }
  // Split segment
  //   0: no unsplit vertex
  //   1: unsplit vertex 0
  //   2: unsplit vertex 1
  //   3: both vertices unsplit (impossible)
  for (PetscInt s = 0; s < 3; ++s) {
    rt         = (DM_POLYTOPE_SEGMENT * 2 + 1) * 100 + s;
    ex->Nt[rt] = 2;
    Nc         = 8 * 2 + 14;
    No         = 2 * 2 + 4;
    PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
    ex->target[rt][0] = DM_POLYTOPE_SEGMENT;
    ex->target[rt][1] = ex->useTensor ? DM_POLYTOPE_SEG_PRISM_TENSOR : DM_POLYTOPE_QUADRILATERAL;
    ex->size[rt][0]   = 2;
    ex->size[rt][1]   = 1;
    //   cones for segments
    for (PetscInt i = 0; i < 2; ++i) {
      ex->cone[rt][8 * i + 0] = DM_POLYTOPE_POINT;
      ex->cone[rt][8 * i + 1] = 1;
      ex->cone[rt][8 * i + 2] = 0;
      ex->cone[rt][8 * i + 3] = s == 1 ? 0 : i;
      ex->cone[rt][8 * i + 4] = DM_POLYTOPE_POINT;
      ex->cone[rt][8 * i + 5] = 1;
      ex->cone[rt][8 * i + 6] = 1;
      ex->cone[rt][8 * i + 7] = s == 2 ? 0 : i;
    }
    for (PetscInt i = 0; i < 2 * 2; ++i) ex->ornt[rt][i] = 0;
    //   cone for quad/tensor quad
    coff = 8 * 2;
    ooff = 2 * 2;
    if (ex->useTensor) {
      ex->cone[rt][coff + 0]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][coff + 1]  = 0;
      ex->cone[rt][coff + 2]  = 0;
      ex->cone[rt][coff + 3]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][coff + 4]  = 0;
      ex->cone[rt][coff + 5]  = 1;
      ex->cone[rt][coff + 6]  = DM_POLYTOPE_POINT_PRISM_TENSOR;
      ex->cone[rt][coff + 7]  = 1;
      ex->cone[rt][coff + 8]  = 0;
      ex->cone[rt][coff + 9]  = 0;
      ex->cone[rt][coff + 10] = DM_POLYTOPE_POINT_PRISM_TENSOR;
      ex->cone[rt][coff + 11] = 1;
      ex->cone[rt][coff + 12] = 1;
      ex->cone[rt][coff + 13] = 0;
      ex->ornt[rt][ooff + 0]  = 0;
      ex->ornt[rt][ooff + 1]  = 0;
      ex->ornt[rt][ooff + 2]  = 0;
      ex->ornt[rt][ooff + 3]  = 0;
    } else {
      ex->cone[rt][coff + 0]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][coff + 1]  = 0;
      ex->cone[rt][coff + 2]  = 0;
      ex->cone[rt][coff + 3]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][coff + 4]  = 1;
      ex->cone[rt][coff + 5]  = 1;
      ex->cone[rt][coff + 6]  = 0;
      ex->cone[rt][coff + 7]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][coff + 8]  = 0;
      ex->cone[rt][coff + 9]  = 1;
      ex->cone[rt][coff + 10] = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][coff + 11] = 1;
      ex->cone[rt][coff + 12] = 0;
      ex->cone[rt][coff + 13] = 0;
      ex->ornt[rt][ooff + 0]  = 0;
      ex->ornt[rt][ooff + 1]  = 0;
      ex->ornt[rt][ooff + 2]  = -1;
      ex->ornt[rt][ooff + 3]  = -1;
    }
  }
  // Impinging segment
  //   0: no splits (impossible)
  //   1: split vertex 0
  //   2: split vertex 1
  //   3: split both vertices (impossible)
  for (PetscInt s = 1; s < 3; ++s) {
    rt         = (DM_POLYTOPE_SEGMENT * 2 + 0) * 100 + s;
    ex->Nt[rt] = 1;
    Nc         = 8;
    No         = 2;
    PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
    ex->target[rt][0] = DM_POLYTOPE_SEGMENT;
    ex->size[rt][0]   = 1;
    //   cone for segment
    ex->cone[rt][0] = DM_POLYTOPE_POINT;
    ex->cone[rt][1] = 1;
    ex->cone[rt][2] = 0;
    ex->cone[rt][3] = s == 1 ? 1 : 0;
    ex->cone[rt][4] = DM_POLYTOPE_POINT;
    ex->cone[rt][5] = 1;
    ex->cone[rt][6] = 1;
    ex->cone[rt][7] = s == 2 ? 1 : 0;
    for (PetscInt i = 0; i < 2; ++i) ex->ornt[rt][i] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DM_POLYTOPE_TRIANGLE produces
     2 triangles, and
     1 triangular prism/tensor triangular prism
*/
static PetscErrorCode DMPlexTransformCohesiveExtrudeSetUp_Triangle(DMPlexTransform_Cohesive *ex)
{
  PetscInt rt, Nc, No, coff, ooff;

  PetscFunctionBegin;
  // No unsplit triangles
  // Split triangles
  //   0: no unsplit edge
  //   1: unsplit edge 0
  //   2: unsplit edge 1
  //   3: unsplit edge 0 1
  //   4: unsplit edge 2
  //   5: unsplit edge 0 2
  //   6: unsplit edge 1 2
  //   7: all edges unsplit (impossible)
  for (PetscInt s = 0; s < 7; ++s) {
    rt         = (DM_POLYTOPE_TRIANGLE * 2 + 1) * 100 + s;
    ex->Nt[rt] = 2;
    Nc         = 12 * 2 + 18;
    No         = 3 * 2 + 5;
    PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
    ex->target[rt][0] = DM_POLYTOPE_TRIANGLE;
    ex->target[rt][1] = ex->useTensor ? DM_POLYTOPE_TRI_PRISM_TENSOR : DM_POLYTOPE_TRI_PRISM;
    ex->size[rt][0]   = 2;
    ex->size[rt][1]   = 1;
    //   cones for triangles
    for (PetscInt i = 0; i < 2; ++i) {
      ex->cone[rt][12 * i + 0]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][12 * i + 1]  = 1;
      ex->cone[rt][12 * i + 2]  = 0;
      ex->cone[rt][12 * i + 3]  = s & 1 ? 0 : i;
      ex->cone[rt][12 * i + 4]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][12 * i + 5]  = 1;
      ex->cone[rt][12 * i + 6]  = 1;
      ex->cone[rt][12 * i + 7]  = s & 2 ? 0 : i;
      ex->cone[rt][12 * i + 8]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][12 * i + 9]  = 1;
      ex->cone[rt][12 * i + 10] = 2;
      ex->cone[rt][12 * i + 11] = s & 4 ? 0 : i;
    }
    for (PetscInt i = 0; i < 3 * 2; ++i) ex->ornt[rt][i] = 0;
    //   cone for triangular prism/tensor triangular prism
    coff = 12 * 2;
    ooff = 3 * 2;
    if (ex->useTensor) {
      ex->cone[rt][coff + 0]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[rt][coff + 1]  = 0;
      ex->cone[rt][coff + 2]  = 0;
      ex->cone[rt][coff + 3]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[rt][coff + 4]  = 0;
      ex->cone[rt][coff + 5]  = 1;
      ex->cone[rt][coff + 6]  = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[rt][coff + 7]  = 1;
      ex->cone[rt][coff + 8]  = 0;
      ex->cone[rt][coff + 9]  = 0;
      ex->cone[rt][coff + 10] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[rt][coff + 11] = 1;
      ex->cone[rt][coff + 12] = 1;
      ex->cone[rt][coff + 13] = 0;
      ex->cone[rt][coff + 14] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[rt][coff + 15] = 1;
      ex->cone[rt][coff + 16] = 2;
      ex->cone[rt][coff + 17] = 0;
      ex->ornt[rt][ooff + 0]  = 0;
      ex->ornt[rt][ooff + 1]  = 0;
      ex->ornt[rt][ooff + 2]  = 0;
      ex->ornt[rt][ooff + 3]  = 0;
      ex->ornt[rt][ooff + 4]  = 0;
    } else {
      ex->cone[rt][coff + 0]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[rt][coff + 1]  = 0;
      ex->cone[rt][coff + 2]  = 0;
      ex->cone[rt][coff + 3]  = DM_POLYTOPE_TRIANGLE;
      ex->cone[rt][coff + 4]  = 0;
      ex->cone[rt][coff + 5]  = 1;
      ex->cone[rt][coff + 6]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 7]  = 1;
      ex->cone[rt][coff + 8]  = 0;
      ex->cone[rt][coff + 9]  = 0;
      ex->cone[rt][coff + 10] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 11] = 1;
      ex->cone[rt][coff + 12] = 1;
      ex->cone[rt][coff + 13] = 0;
      ex->cone[rt][coff + 14] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 15] = 1;
      ex->cone[rt][coff + 16] = 2;
      ex->cone[rt][coff + 17] = 0;
      ex->ornt[rt][ooff + 0]  = -2;
      ex->ornt[rt][ooff + 1]  = 0;
      ex->ornt[rt][ooff + 2]  = 0;
      ex->ornt[rt][ooff + 3]  = 0;
      ex->ornt[rt][ooff + 4]  = 0;
    }
  }
  // Impinging triangles
  //   0: no splits (impossible)
  //   1: split edge 0
  //   2: split edge 1
  //   3: split edges 0 and 1
  //   4: split edge 2
  //   5: split edges 0 and 2
  //   6: split edges 1 and 2
  //   7: split all edges (impossible)
  for (PetscInt s = 1; s < 7; ++s) {
    rt         = (DM_POLYTOPE_TRIANGLE * 2 + 0) * 100 + s;
    ex->Nt[rt] = 1;
    Nc         = 12;
    No         = 3;
    PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
    ex->target[rt][0] = DM_POLYTOPE_TRIANGLE;
    ex->size[rt][0]   = 1;
    //   cone for triangle
    ex->cone[rt][0]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][1]  = 1;
    ex->cone[rt][2]  = 0;
    ex->cone[rt][3]  = s & 1 ? 1 : 0;
    ex->cone[rt][4]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][5]  = 1;
    ex->cone[rt][6]  = 1;
    ex->cone[rt][7]  = s & 2 ? 1 : 0;
    ex->cone[rt][8]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][9]  = 1;
    ex->cone[rt][10] = 2;
    ex->cone[rt][11] = s & 4 ? 1 : 0;
    for (PetscInt i = 0; i < 3; ++i) ex->ornt[rt][i] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* DM_POLYTOPE_QUADRILATERAL produces
     2 quads, and
     1 hex/tensor hex
*/
static PetscErrorCode DMPlexTransformCohesiveExtrudeSetUp_Quadrilateral(DMPlexTransform_Cohesive *ex)
{
  PetscInt rt, Nc, No, coff, ooff;

  PetscFunctionBegin;
  // No unsplit quadrilaterals
  // Split quadrilateral
  //   0: no unsplit edge
  //   1: unsplit edge 0
  //   2: unsplit edge 1
  //   3: unsplit edge 0 1
  //   4: unsplit edge 2
  //   5: unsplit edge 0 2
  //   6: unsplit edge 1 2
  //   7: unsplit edge 0 1 2
  //   8: unsplit edge 3
  //   9: unsplit edge 0 3
  //  10: unsplit edge 1 3
  //  11: unsplit edge 0 1 3
  //  12: unsplit edge 2 3
  //  13: unsplit edge 0 2 3
  //  14: unsplit edge 1 2 3
  //  15: all edges unsplit (impossible)
  for (PetscInt s = 0; s < 15; ++s) {
    rt         = (DM_POLYTOPE_QUADRILATERAL * 2 + 1) * 100 + s;
    ex->Nt[rt] = 2;
    Nc         = 16 * 2 + 22;
    No         = 4 * 2 + 6;
    PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
    ex->target[rt][0] = DM_POLYTOPE_QUADRILATERAL;
    ex->target[rt][1] = ex->useTensor ? DM_POLYTOPE_QUAD_PRISM_TENSOR : DM_POLYTOPE_HEXAHEDRON;
    ex->size[rt][0]   = 2;
    ex->size[rt][1]   = 1;
    //   cones for quads
    for (PetscInt i = 0; i < 2; ++i) {
      ex->cone[rt][16 * i + 0]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][16 * i + 1]  = 1;
      ex->cone[rt][16 * i + 2]  = 0;
      ex->cone[rt][16 * i + 3]  = s & 1 ? 0 : i;
      ex->cone[rt][16 * i + 4]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][16 * i + 5]  = 1;
      ex->cone[rt][16 * i + 6]  = 1;
      ex->cone[rt][16 * i + 7]  = s & 2 ? 0 : i;
      ex->cone[rt][16 * i + 8]  = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][16 * i + 9]  = 1;
      ex->cone[rt][16 * i + 10] = 2;
      ex->cone[rt][16 * i + 11] = s & 4 ? 0 : i;
      ex->cone[rt][16 * i + 12] = DM_POLYTOPE_SEGMENT;
      ex->cone[rt][16 * i + 13] = 1;
      ex->cone[rt][16 * i + 14] = 3;
      ex->cone[rt][16 * i + 15] = s & 8 ? 0 : i;
    }
    for (PetscInt i = 0; i < 4 * 2; ++i) ex->ornt[rt][i] = 0;
    //   cones for hexes/tensor hexes
    coff = 16 * 2;
    ooff = 4 * 2;
    if (ex->useTensor) {
      ex->cone[rt][coff + 0]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 1]  = 0;
      ex->cone[rt][coff + 2]  = 0;
      ex->cone[rt][coff + 3]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 4]  = 0;
      ex->cone[rt][coff + 5]  = 1;
      ex->cone[rt][coff + 6]  = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[rt][coff + 7]  = 1;
      ex->cone[rt][coff + 8]  = 0;
      ex->cone[rt][coff + 9]  = 0;
      ex->cone[rt][coff + 10] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[rt][coff + 11] = 1;
      ex->cone[rt][coff + 12] = 1;
      ex->cone[rt][coff + 13] = 0;
      ex->cone[rt][coff + 14] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[rt][coff + 15] = 1;
      ex->cone[rt][coff + 16] = 2;
      ex->cone[rt][coff + 17] = 0;
      ex->cone[rt][coff + 18] = DM_POLYTOPE_SEG_PRISM_TENSOR;
      ex->cone[rt][coff + 19] = 1;
      ex->cone[rt][coff + 20] = 3;
      ex->cone[rt][coff + 21] = 0;
      ex->ornt[rt][ooff + 0]  = 0;
      ex->ornt[rt][ooff + 1]  = 0;
      ex->ornt[rt][ooff + 2]  = 0;
      ex->ornt[rt][ooff + 3]  = 0;
      ex->ornt[rt][ooff + 4]  = 0;
      ex->ornt[rt][ooff + 5]  = 0;
    } else {
      ex->cone[rt][coff + 0]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 1]  = 0;
      ex->cone[rt][coff + 2]  = 0;
      ex->cone[rt][coff + 3]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 4]  = 0;
      ex->cone[rt][coff + 5]  = 1;
      ex->cone[rt][coff + 6]  = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 7]  = 1;
      ex->cone[rt][coff + 8]  = 0;
      ex->cone[rt][coff + 9]  = 0;
      ex->cone[rt][coff + 10] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 11] = 1;
      ex->cone[rt][coff + 12] = 2;
      ex->cone[rt][coff + 13] = 0;
      ex->cone[rt][coff + 14] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 15] = 1;
      ex->cone[rt][coff + 16] = 1;
      ex->cone[rt][coff + 17] = 0;
      ex->cone[rt][coff + 18] = DM_POLYTOPE_QUADRILATERAL;
      ex->cone[rt][coff + 19] = 1;
      ex->cone[rt][coff + 20] = 3;
      ex->cone[rt][coff + 21] = 0;
      ex->ornt[rt][ooff + 0]  = -2;
      ex->ornt[rt][ooff + 1]  = 0;
      ex->ornt[rt][ooff + 2]  = 0;
      ex->ornt[rt][ooff + 3]  = 0;
      ex->ornt[rt][ooff + 4]  = 0;
      ex->ornt[rt][ooff + 5]  = 1;
    }
  }
  // Impinging quadrilaterals
  //   0:  no splits (impossible)
  //   1:  split edge 0
  //   2:  split edge 1
  //   3:  split edges 0 and 1
  //   4:  split edge 2
  //   5:  split edges 0 and 2
  //   6:  split edges 1 and 2
  //   7:  split edges 0, 1, and 2
  //   8:  split edge 3
  //   9:  split edges 0 and 3
  //   10: split edges 1 and 3
  //   11: split edges 0, 1, and 3
  //   12: split edges 2 and 3
  //   13: split edges 0, 2, and 3
  //   14: split edges 1, 2, and 3
  //   15: split all edges (impossible)
  for (PetscInt s = 1; s < 15; ++s) {
    rt         = (DM_POLYTOPE_QUADRILATERAL * 2 + 0) * 100 + s;
    ex->Nt[rt] = 1;
    Nc         = 16;
    No         = 4;
    PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
    ex->target[rt][0] = DM_POLYTOPE_QUADRILATERAL;
    ex->size[rt][0]   = 1;
    //   cone for quadrilateral
    ex->cone[rt][0]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][1]  = 1;
    ex->cone[rt][2]  = 0;
    ex->cone[rt][3]  = s & 1 ? 1 : 0;
    ex->cone[rt][4]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][5]  = 1;
    ex->cone[rt][6]  = 1;
    ex->cone[rt][7]  = s & 2 ? 1 : 0;
    ex->cone[rt][8]  = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][9]  = 1;
    ex->cone[rt][10] = 2;
    ex->cone[rt][11] = s & 4 ? 1 : 0;
    ex->cone[rt][12] = DM_POLYTOPE_SEGMENT;
    ex->cone[rt][13] = 1;
    ex->cone[rt][14] = 3;
    ex->cone[rt][15] = s & 8 ? 1 : 0;
    for (PetscInt i = 0; i < 4; ++i) ex->ornt[rt][i] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformCohesiveExtrudeSetUp_Tetrahedron(DMPlexTransform_Cohesive *ex)
{
  PetscInt rt, Nc, No;

  PetscFunctionBegin;
  // Impinging tetrahedra
  //   0:  no splits (impossible)
  //   1:  split face 0
  //   2:  split face 1
  //   3:  split faces 0 and 1
  //   4:  split face 2
  //   5:  split faces 0 and 2
  //   6:  split faces 1 and 2
  //   7:  split faces 0, 1, and 2
  //   8:  split face 3
  //   9:  split faces 0 and 3
  //   10: split faces 1 and 3
  //   11: split faces 0, 1, and 3
  //   12: split faces 2 and 3
  //   13: split faces 0, 2, and 3
  //   14: split faces 1, 2, and 3
  //   15: split all faces (impossible)
  for (PetscInt s = 1; s < 15; ++s) {
    rt         = (DM_POLYTOPE_TETRAHEDRON * 2 + 0) * 100 + s;
    ex->Nt[rt] = 1;
    Nc         = 16;
    No         = 4;
    PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
    ex->target[rt][0] = DM_POLYTOPE_TETRAHEDRON;
    ex->size[rt][0]   = 1;
    //   cone for triangle
    ex->cone[rt][0]  = DM_POLYTOPE_TRIANGLE;
    ex->cone[rt][1]  = 1;
    ex->cone[rt][2]  = 0;
    ex->cone[rt][3]  = s & 1 ? 1 : 0;
    ex->cone[rt][4]  = DM_POLYTOPE_TRIANGLE;
    ex->cone[rt][5]  = 1;
    ex->cone[rt][6]  = 1;
    ex->cone[rt][7]  = s & 2 ? 1 : 0;
    ex->cone[rt][8]  = DM_POLYTOPE_TRIANGLE;
    ex->cone[rt][9]  = 1;
    ex->cone[rt][10] = 2;
    ex->cone[rt][11] = s & 4 ? 1 : 0;
    ex->cone[rt][12] = DM_POLYTOPE_TRIANGLE;
    ex->cone[rt][13] = 1;
    ex->cone[rt][14] = 3;
    ex->cone[rt][15] = s & 8 ? 1 : 0;
    for (PetscInt i = 0; i < 4; ++i) ex->ornt[rt][i] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformCohesiveExtrudeSetUp_Hexahedron(DMPlexTransform_Cohesive *ex)
{
  PetscInt rt, Nc, No;

  PetscFunctionBegin;
  // Impinging hexahedra
  //   0:  no splits (impossible)
  //   bit is set if the face is split
  //   63: split all faces (impossible)
  for (PetscInt s = 1; s < 63; ++s) {
    rt         = (DM_POLYTOPE_HEXAHEDRON * 2 + 0) * 100 + s;
    ex->Nt[rt] = 1;
    Nc         = 24;
    No         = 6;
    PetscCall(PetscMalloc4(ex->Nt[rt], &ex->target[rt], ex->Nt[rt], &ex->size[rt], Nc, &ex->cone[rt], No, &ex->ornt[rt]));
    ex->target[rt][0] = DM_POLYTOPE_HEXAHEDRON;
    ex->size[rt][0]   = 1;
    //   cone for hexahedron
    ex->cone[rt][0]  = DM_POLYTOPE_QUADRILATERAL;
    ex->cone[rt][1]  = 1;
    ex->cone[rt][2]  = 0;
    ex->cone[rt][3]  = s & 1 ? 1 : 0;
    ex->cone[rt][4]  = DM_POLYTOPE_QUADRILATERAL;
    ex->cone[rt][5]  = 1;
    ex->cone[rt][6]  = 1;
    ex->cone[rt][7]  = s & 2 ? 1 : 0;
    ex->cone[rt][8]  = DM_POLYTOPE_QUADRILATERAL;
    ex->cone[rt][9]  = 1;
    ex->cone[rt][10] = 2;
    ex->cone[rt][11] = s & 4 ? 1 : 0;
    ex->cone[rt][12] = DM_POLYTOPE_QUADRILATERAL;
    ex->cone[rt][13] = 1;
    ex->cone[rt][14] = 3;
    ex->cone[rt][15] = s & 8 ? 1 : 0;
    ex->cone[rt][16] = DM_POLYTOPE_QUADRILATERAL;
    ex->cone[rt][17] = 1;
    ex->cone[rt][18] = 4;
    ex->cone[rt][19] = s & 16 ? 1 : 0;
    ex->cone[rt][20] = DM_POLYTOPE_QUADRILATERAL;
    ex->cone[rt][21] = 1;
    ex->cone[rt][22] = 5;
    ex->cone[rt][23] = s & 32 ? 1 : 0;
    for (PetscInt i = 0; i < 6; ++i) ex->ornt[rt][i] = 0;
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The refine types for cohesive extrusion are:

  ct * 2 + 0:                  For any point which should just return itself
  ct * 2 + 1:                  For unsplit points
  (ct * 2 + 0) * 100 + fsplit: For impinging points, one type for each combination of split faces
  (ct * 2 + 1) * 100 + fsplit: For split points, one type for each combination of unsplit faces
*/
static PetscErrorCode DMPlexTransformSetUp_Cohesive(DMPlexTransform tr)
{
  DMPlexTransform_Cohesive *ex = (DMPlexTransform_Cohesive *)tr->data;
  DM                        dm;
  DMLabel                   active, celltype;
  PetscInt                  numRt, pStart, pEnd, ict;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMPlexTransformGetActive(tr, &active));
  PetscCheck(active, PetscObjectComm((PetscObject)tr), PETSC_ERR_ARG_WRONG, "Cohesive extrusion requires an active label");
  PetscCall(DMPlexGetCellTypeLabel(dm, &celltype));
  PetscCall(DMLabelCreate(PETSC_COMM_SELF, "Refine Type", &tr->trType));
  PetscCall(DMPlexGetChart(dm, &pStart, &pEnd));
  PetscCall(DMLabelMakeAllInvalid_Internal(active));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    PetscInt ct, val;

    PetscCall(DMLabelGetValue(celltype, p, &ct));
    PetscCall(DMLabelGetValue(active, p, &val));
    if (val < 0) {
      // Also negative size impinging points
      // ct * 2 + 0 is the identity transform
      PetscCall(DMLabelSetValue(tr->trType, p, ct * 2 + 0));
    } else {
      PetscInt fsplit = -1, funsplit = -1;

      // Unsplit points ct * 2 + 1
      if (val >= 200) {
        // Cohesive cells cannot be unsplit
        //   This is faulty inheritance through the label
        if (ct == DM_POLYTOPE_POINT_PRISM_TENSOR || ct == DM_POLYTOPE_SEG_PRISM_TENSOR || ct == DM_POLYTOPE_TRI_PRISM_TENSOR || ct == DM_POLYTOPE_QUAD_PRISM_TENSOR) PetscCall(DMLabelSetValue(tr->trType, p, ct * 2 + 0));
        else PetscCall(DMLabelSetValue(tr->trType, p, ct * 2 + 1));
      } else if (val >= 100) {
        // Impinging points: (ct * 2 + 0) * 100 + fsplit
        PetscCall(ComputeSplitFaceNumber(dm, active, p, &fsplit));
        if (!fsplit) PetscCall(DMLabelSetValue(tr->trType, p, ct * 2 + 0));
        else PetscCall(DMLabelSetValue(tr->trType, p, (ct * 2 + 0) * 100 + fsplit));
      } else {
        // Split points: (ct * 2 + 1) * 100 + funsplit
        PetscCall(ComputeUnsplitFaceNumber(dm, active, p, &funsplit));
        PetscCall(DMLabelSetValue(tr->trType, p, (ct * 2 + 1) * 100 + funsplit));
      }
    }
  }
  if (ex->debug) {
    PetscCall(DMLabelView(active, NULL));
    PetscCall(DMLabelView(tr->trType, NULL));
  }
  numRt = DM_NUM_POLYTOPES * 2 * 100;
  PetscCall(PetscMalloc5(numRt, &ex->Nt, numRt, &ex->target, numRt, &ex->size, numRt, &ex->cone, numRt, &ex->ornt));
  for (ict = 0; ict < numRt; ++ict) {
    ex->Nt[ict]     = -1;
    ex->target[ict] = NULL;
    ex->size[ict]   = NULL;
    ex->cone[ict]   = NULL;
    ex->ornt[ict]   = NULL;
  }
  PetscCall(DMPlexTransformCohesiveExtrudeSetUp_Point(ex));
  PetscCall(DMPlexTransformCohesiveExtrudeSetUp_Segment(ex));
  PetscCall(DMPlexTransformCohesiveExtrudeSetUp_Triangle(ex));
  PetscCall(DMPlexTransformCohesiveExtrudeSetUp_Quadrilateral(ex));
  PetscCall(DMPlexTransformCohesiveExtrudeSetUp_Tetrahedron(ex));
  PetscCall(DMPlexTransformCohesiveExtrudeSetUp_Hexahedron(ex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformDestroy_Cohesive(DMPlexTransform tr)
{
  DMPlexTransform_Cohesive *ex = (DMPlexTransform_Cohesive *)tr->data;
  PetscInt                  ct;

  PetscFunctionBegin;
  if (ex->target) {
    for (ct = 0; ct < DM_NUM_POLYTOPES * 2 * 100; ++ct) PetscCall(PetscFree4(ex->target[ct], ex->size[ct], ex->cone[ct], ex->ornt[ct]));
  }
  PetscCall(PetscFree5(ex->Nt, ex->target, ex->size, ex->cone, ex->ornt));
  PetscCall(PetscFree(ex));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_Cohesive(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  DMPlexTransform_Cohesive *ex     = (DMPlexTransform_Cohesive *)tr->data;
  DMLabel                   trType = tr->trType;
  PetscInt                  rt;

  PetscFunctionBeginHot;
  *rnew = r;
  *onew = DMPolytopeTypeComposeOrientation(tct, o, so);
  if (!so) PetscFunctionReturn(PETSC_SUCCESS);
  if (trType) {
    PetscCall(DMLabelGetValue(tr->trType, sp, &rt));
    if (rt < 100 && !(rt % 2)) PetscFunctionReturn(PETSC_SUCCESS);
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
        *onew = DMPolytopeTypeComposeOrientation(tct, o, so ? -1 : 0);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
    // We need to handle identity extrusions from volumes (TET, HEX, etc) when boundary faces are being extruded
    case DM_POLYTOPE_TRIANGLE:
      break;
    case DM_POLYTOPE_QUADRILATERAL:
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
        *onew = DMPolytopeTypeComposeOrientation(tct, o, so ? -3 : 0);
        break;
      default:
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
    // We need to handle identity extrusions from volumes (TET, HEX, etc) when boundary faces are being extruded
    case DM_POLYTOPE_TRIANGLE:
      break;
    case DM_POLYTOPE_QUADRILATERAL:
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell type %s", DMPolytopeTypes[sct]);
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformCellTransform_Cohesive(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DMPlexTransform_Cohesive *ex       = (DMPlexTransform_Cohesive *)tr->data;
  DMLabel                   trType   = tr->trType;
  PetscBool                 identity = PETSC_FALSE;
  PetscInt                  val      = 0;

  PetscFunctionBegin;
  PetscCheck(trType, PETSC_COMM_SELF, PETSC_ERR_SUP, "Missing transform type label");
  PetscCall(DMLabelGetValue(trType, p, &val));
  identity = val < 100 && !(val % 2) ? PETSC_TRUE : PETSC_FALSE;
  if (rt) *rt = val;
  if (identity) {
    PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, NULL, Nt, target, size, cone, ornt));
  } else {
    *Nt     = ex->Nt[val];
    *target = ex->target[val];
    *size   = ex->size[val];
    *cone   = ex->cone[val];
    *ornt   = ex->ornt[val];
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode OrderCohesiveSupport_Private(DM dm, PetscInt p)
{
  const PetscInt *cone;
  PetscInt        csize;

  PetscFunctionBegin;
  PetscCall(DMPlexGetCone(dm, p, &cone));
  PetscCall(DMPlexGetConeSize(dm, p, &csize));
  PetscCheck(csize > 2, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cone size for cohesive cell should be > 2, not %" PetscInt_FMT, csize);
  for (PetscInt s = 0; s < 2; ++s) {
    const PetscInt *supp;
    PetscInt        Ns, neighbor;

    PetscCall(DMPlexGetSupport(dm, cone[s], &supp));
    PetscCall(DMPlexGetSupportSize(dm, cone[s], &Ns));
    // Could check here that the face is in the pointSF
    if (Ns == 1) continue;
    PetscCheck(Ns == 2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cohesive cell endcap should have support of size 2, not %" PetscInt_FMT, Ns);
    PetscCheck((supp[0] == p) != (supp[1] == p), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cohesive cell %" PetscInt_FMT " endcap must have cell in support once", p);
    neighbor = supp[s] == p ? supp[1 - s] : -1;
    if (neighbor >= 0) {
      PetscCall(DMPlexInsertSupport(dm, cone[s], s, neighbor));
      PetscCall(DMPlexInsertSupport(dm, cone[s], 1 - s, p));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// We need the supports of endcap faces on cohesive cells to have the same orientation
//   We cannot just fix split points, since we destroy support ordering with DMPLexSymmetrize()
static PetscErrorCode DMPlexTransformOrderSupports_Cohesive(DMPlexTransform tr, DM dm, DM tdm)
{
  PetscInt dim, pStart, pEnd;

  PetscFunctionBegin;
  PetscCall(DMGetDimension(dm, &dim));
  PetscCall(DMPlexGetChart(tdm, &pStart, &pEnd));
  for (PetscInt p = pStart; p < pEnd; ++p) {
    DMPolytopeType ct;

    PetscCall(DMPlexGetCellType(tdm, p, &ct));
    switch (dim) {
    case 2:
      if (ct == DM_POLYTOPE_SEG_PRISM_TENSOR) PetscCall(OrderCohesiveSupport_Private(tdm, p));
      break;
    case 3:
      if (ct == DM_POLYTOPE_TRI_PRISM_TENSOR || ct == DM_POLYTOPE_QUAD_PRISM_TENSOR) PetscCall(OrderCohesiveSupport_Private(tdm, p));
      break;
    default:
      break;
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

/* New vertices have the same coordinates */
static PetscErrorCode DMPlexTransformMapCoordinates_Cohesive(DMPlexTransform tr, DMPolytopeType pct, DMPolytopeType ct, PetscInt p, PetscInt r, PetscInt Nv, PetscInt dE, const PetscScalar in[], PetscScalar out[])
{
  PetscReal width;
  PetscInt  pval;

  PetscFunctionBeginHot;
  PetscCheck(pct == DM_POLYTOPE_POINT, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for parent point type %s", DMPolytopeTypes[pct]);
  PetscCheck(ct == DM_POLYTOPE_POINT, PETSC_COMM_SELF, PETSC_ERR_SUP, "Not for refined point type %s", DMPolytopeTypes[ct]);
  PetscCheck(Nv == 1, PETSC_COMM_SELF, PETSC_ERR_SUP, "Vertices should be produced from a single vertex, not %" PetscInt_FMT, Nv);
  PetscCheck(r < 2, PETSC_COMM_SELF, PETSC_ERR_SUP, "Vertices should only have two replicas, not %" PetscInt_FMT, r);

  PetscCall(DMPlexTransformCohesiveExtrudeGetWidth(tr, &width));
  PetscCall(DMLabelGetValue(tr->trType, p, &pval));
  if (width == 0. || pval < 100) {
    for (PetscInt d = 0; d < dE; ++d) out[d] = in[d];
  } else {
    DM        dm;
    PetscReal avgNormal[3] = {0., 0., 0.}, norm = 0.;
    PetscInt *star = NULL;
    PetscInt  Nst, fStart, fEnd, Nf = 0;

    PetscCall(DMPlexTransformGetDM(tr, &dm));
    PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
    PetscCall(DMPlexGetTransitiveClosure(dm, p, PETSC_FALSE, &Nst, &star));
    // Get support faces that are split, refine type (ct * 2 + 1) * 100 + fsplit
    for (PetscInt st = 0; st < Nst * 2; st += 2) {
      DMPolytopeType ct;
      PetscInt       val;

      if (star[st] < fStart || star[st] >= fEnd) continue;
      PetscCall(DMPlexGetCellType(dm, star[st], &ct));
      PetscCall(DMLabelGetValue(tr->trType, star[st], &val));
      if (val < ((PetscInt)ct * 2 + 1) * 100) continue;
      star[Nf++] = star[st];
    }
    PetscCheck(Nf, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Split vertex %" PetscInt_FMT " must be connected to at least one split face", p);
    // Average normals
    for (PetscInt f = 0; f < Nf; ++f) {
      PetscReal normal[3], vol;

      PetscCall(DMPlexComputeCellGeometryFVM(dm, star[f], &vol, NULL, normal));
      for (PetscInt d = 0; d < dE; ++d) avgNormal[d] += normal[d];
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, p, PETSC_FALSE, &Nst, &star));
    // Normalize normal
    for (PetscInt d = 0; d < dE; ++d) norm += PetscSqr(avgNormal[d]);
    norm = PetscSqrtReal(norm);
    for (PetscInt d = 0; d < dE; ++d) avgNormal[d] /= norm;
    // Symmetrically push vertices along normal
    for (PetscInt d = 0; d < dE; ++d) out[d] = in[d] + width * avgNormal[d] * (r ? -0.5 : 0.5);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformInitialize_Cohesive(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                  = DMPlexTransformView_Cohesive;
  tr->ops->setfromoptions        = DMPlexTransformSetFromOptions_Cohesive;
  tr->ops->setup                 = DMPlexTransformSetUp_Cohesive;
  tr->ops->destroy               = DMPlexTransformDestroy_Cohesive;
  tr->ops->setdimensions         = DMPlexTransformSetDimensions_Internal;
  tr->ops->celltransform         = DMPlexTransformCellTransform_Cohesive;
  tr->ops->ordersupports         = DMPlexTransformOrderSupports_Cohesive;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_Cohesive;
  tr->ops->mapcoordinates        = DMPlexTransformMapCoordinates_Cohesive;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Cohesive(DMPlexTransform tr)
{
  DMPlexTransform_Cohesive *ex;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&ex));
  tr->data      = ex;
  ex->useTensor = PETSC_TRUE;
  PetscCall(DMPlexTransformInitialize_Cohesive(tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformCohesiveExtrudeGetTensor - Get the flag to use tensor cells

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

.seealso: `DMPlexTransform`, `DMPlexTransformCohesiveExtrudeSetTensor()`, `DMPlexTransformExtrudeGetTensor()`
@*/
PetscErrorCode DMPlexTransformCohesiveExtrudeGetTensor(DMPlexTransform tr, PetscBool *useTensor)
{
  DMPlexTransform_Cohesive *ex = (DMPlexTransform_Cohesive *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(useTensor, 2);
  *useTensor = ex->useTensor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformCohesiveExtrudeSetTensor - Set the flag to use tensor cells

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

.seealso: `DMPlexTransform`, `DMPlexTransformCohesiveExtrudeGetTensor()`, `DMPlexTransformExtrudeSetTensor()`
@*/
PetscErrorCode DMPlexTransformCohesiveExtrudeSetTensor(DMPlexTransform tr, PetscBool useTensor)
{
  DMPlexTransform_Cohesive *ex = (DMPlexTransform_Cohesive *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->useTensor = useTensor;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformCohesiveExtrudeGetWidth - Get the width of extruded cells

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. width - The width of extruded cells, or 0.

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformCohesiveExtrudeSetWidth()`
@*/
PetscErrorCode DMPlexTransformCohesiveExtrudeGetWidth(DMPlexTransform tr, PetscReal *width)
{
  DMPlexTransform_Cohesive *ex = (DMPlexTransform_Cohesive *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(width, 2);
  *width = ex->width;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformCohesiveExtrudeSetWidth - Set the width of extruded cells

  Not Collective

  Input Parameters:
+ tr    - The `DMPlexTransform`
- width - The width of the extruded cells, or 0.

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexTransformCohesiveExtrudeGetWidth()`
@*/
PetscErrorCode DMPlexTransformCohesiveExtrudeSetWidth(DMPlexTransform tr, PetscReal width)
{
  DMPlexTransform_Cohesive *ex = (DMPlexTransform_Cohesive *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ex->width = width;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexTransformCohesiveExtrudeGetUnsplit - Get a new label marking the unsplit points in the transformed mesh

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. unsplit - A new `DMLabel` marking the unsplit points in the transformed mesh

  Level: intermediate

  Note:
  This label should be destroyed by the caller.

.seealso: `DMPlexTransform`, `DMPlexTransformGetTransformTypes()`
@*/
PetscErrorCode DMPlexTransformCohesiveExtrudeGetUnsplit(DMPlexTransform tr, DMLabel *unsplit)
{
  DM              dm;
  DMLabel         trTypes;
  IS              valueIS;
  const PetscInt *values;
  PetscInt        Nv;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(unsplit, 2);
  PetscCall(DMLabelCreate(PetscObjectComm((PetscObject)tr), "Unsplit Points", unsplit));
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMPlexTransformGetTransformTypes(tr, &trTypes));
  PetscCall(DMLabelGetValueIS(trTypes, &valueIS));
  PetscCall(ISGetLocalSize(valueIS, &Nv));
  PetscCall(ISGetIndices(valueIS, &values));
  for (PetscInt v = 0; v < Nv; ++v) {
    const PetscInt  val = values[v];
    IS              pointIS;
    const PetscInt *points;
    PetscInt        Np;

    if (val > 2 * DM_NUM_POLYTOPES || !(val % 2)) continue;
    PetscCall(DMLabelGetStratumIS(trTypes, val, &pointIS));
    PetscCall(ISGetLocalSize(pointIS, &Np));
    PetscCall(ISGetIndices(pointIS, &points));
    for (PetscInt p = 0; p < Np; ++p) {
      const PetscInt  point = points[p];
      DMPolytopeType  ct;
      DMPolytopeType *rct;
      PetscInt       *rsize, *rcone, *rornt;
      PetscInt        Nct, pNew = 0;

      PetscCall(DMPlexGetCellType(dm, point, &ct));
      PetscCall(DMPlexTransformCellTransform(tr, ct, point, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
      for (PetscInt n = 0; n < Nct; ++n) {
        for (PetscInt r = 0; r < rsize[n]; ++r) {
          PetscCall(DMPlexTransformGetTargetPoint(tr, ct, rct[n], point, r, &pNew));
          PetscCall(DMLabelSetValue(*unsplit, pNew, val + tr->labelReplicaInc * r));
        }
      }
    }
    PetscCall(ISRestoreIndices(pointIS, &points));
    PetscCall(ISDestroy(&pointIS));
  }
  PetscCall(ISRestoreIndices(valueIS, &values));
  PetscCall(ISDestroy(&valueIS));
  PetscFunctionReturn(PETSC_SUCCESS);
}
