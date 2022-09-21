#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformView_Alfeld(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "Alfeld refinement %s\n", name ? name : ""));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetUp_Alfeld(DMPlexTransform tr)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_Alfeld(DMPlexTransform tr)
{
  DMPlexRefine_Alfeld *f = (DMPlexRefine_Alfeld *)tr->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(f));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_Alfeld(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  DM              dm;
  PetscInt        dim;
  static PetscInt tri_seg[] = {1, 0, 0, 0, 2, 0, 0, 0, 2, 0, 1, 0, 2, 0, 1, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0};
  static PetscInt tri_tri[] = {0, -3, 2, -3, 1, -3, 2, -3, 1, -3, 0, -3, 1, -3, 0, -3, 2, -3, 0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 0, 0, 2, 0, 0, 0, 1, 0};
  static PetscInt tet_seg[] = {2, 0, 3, 0, 1, 0, 0, 0, 3, 0, 1, 0, 2, 0, 0, 0, 1, 0, 2, 0, 3, 0, 0, 0, 3, 0, 2, 0, 0, 0, 1, 0, 2, 0, 0, 0, 3, 0, 1, 0, 0, 0, 3, 0, 2, 0, 1, 0, 0, 0, 1, 0, 3, 0, 2, 0, 1, 0, 3, 0, 0, 0, 2, 0,
                               3, 0, 0, 0, 1, 0, 2, 0, 1, 0, 0, 0, 2, 0, 3, 0, 0, 0, 2, 0, 1, 0, 3, 0, 2, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1, 0, 2, 0, 3, 0, 1, 0, 2, 0, 0, 0, 3, 0, 2, 0, 0, 0, 1, 0, 3, 0, 1, 0, 0, 0, 3, 0, 2, 0,
                               0, 0, 3, 0, 1, 0, 2, 0, 3, 0, 1, 0, 0, 0, 2, 0, 2, 0, 3, 0, 0, 0, 1, 0, 3, 0, 0, 0, 2, 0, 1, 0, 0, 0, 2, 0, 3, 0, 1, 0, 3, 0, 2, 0, 1, 0, 0, 0, 2, 0, 1, 0, 3, 0, 0, 0, 1, 0, 3, 0, 2, 0, 0, 0};
  static PetscInt tet_tri[] = {5, 1,  2, 0,  4, 0,  1, 1,  3, 2,  0, -2, 4, 1,  5, 0,  2, 0,  3, -1, 0, 2,  1, 0,  2, 1,  4, 0,  5, 0,  0, -1, 1, -3, 3, -2, 5, -2, 3, 2,  1, 0,  4, 1,  2, 0,  0, 2,  1, 1,  5, -3, 3, 2,  2, -2, 0, -2,
                               4, 0,  3, 0,  1, 0,  5, -3, 0, 0,  4, -3, 2, -3, 0, 0,  3, -2, 4, -3, 1, -2, 2, -3, 5, -3, 4, -2, 0, 2,  3, -2, 2, 1,  5, 0,  1, -3, 3, -1, 4, -3, 0, 2,  5, -2, 1, 0,  2, 0,  0, -1, 2, -3, 1, -3, 4, -2,
                               3, -2, 5, 0,  1, -2, 0, -2, 2, -3, 3, 0,  5, -3, 4, -3, 2, -2, 1, -3, 0, -2, 5, 1,  4, 0,  3, 2,  0, 0,  1, 0,  2, 0,  3, 0,  4, 0,  5, 0,  2, 1,  0, 2,  1, 0,  4, -2, 5, -3, 3, 2,  1, 1,  2, 0,  0, 2,
                               5, 1,  3, -2, 4, -3, 0, -1, 4, 0,  3, 2,  2, 1,  1, 0,  5, -3, 3, 0,  0, -2, 4, 0,  1, -2, 5, 0,  2, 0,  4, 1,  3, 2,  0, -2, 5, -2, 2, -3, 1, -3, 5, 1,  1, -3, 3, -2, 2, -2, 4, -3, 0, 2,  3, -1, 5, 0,
                               1, -3, 4, 1,  0, -2, 2, -3, 1, -2, 3, -2, 5, 0,  0, 0,  2, 0,  4, 0,  5, -2, 4, -3, 2, -3, 3, -1, 1, -3, 0, -2, 2, -2, 5, -3, 4, -3, 1, 1,  0, 2,  3, -2, 4, -2, 2, -3, 5, -3, 0, -1, 3, 2,  1, 0};
  static PetscInt tet_tet[] = {3, -2, 2, -3, 0, -1, 1, -1, 3, -1, 1, -3, 2, -1, 0, -1, 3, -3, 0, -3, 1, -1, 2, -1, 2, -1, 3, -1, 1, -3, 0, -2, 2, -3, 0, -1, 3, -2, 1, -3, 2, -2, 1, -2, 0, -2, 3, -2,
                               1, -2, 0, -2, 2, -2, 3, -1, 1, -1, 3, -3, 0, -3, 2, -2, 1, -3, 2, -1, 3, -1, 0, -3, 0, -3, 1, -1, 3, -3, 2, -3, 0, -2, 2, -2, 1, -2, 3, -3, 0, -1, 3, -2, 2, -3, 1, -2,
                               0, 0,  1, 0,  2, 0,  3, 0,  0, 1,  3, 1,  1, 2,  2, 0,  0, 2,  2, 1,  3, 0,  1, 2,  1, 2,  0, 1,  3, 1,  2, 2,  1, 0,  2, 0,  0, 0,  3, 1,  1, 1,  3, 2,  2, 2,  0, 0,
                               2, 1,  3, 0,  0, 2,  1, 0,  2, 2,  1, 1,  3, 2,  0, 2,  2, 0,  0, 0,  1, 0,  3, 2,  3, 2,  2, 2,  1, 1,  0, 1,  3, 0,  0, 2,  2, 1,  1, 1,  3, 1,  1, 2,  0, 1,  2, 1};

  PetscFunctionBeginHot;
  *rnew = r;
  *onew = o;
  if (!so) PetscFunctionReturn(0);
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  if (dim == 2 && sct == DM_POLYTOPE_TRIANGLE) {
    switch (tct) {
    case DM_POLYTOPE_POINT:
      break;
    case DM_POLYTOPE_SEGMENT:
      *rnew = tri_seg[(so + 3) * 6 + r * 2];
      *onew = DMPolytopeTypeComposeOrientation(tct, o, tri_seg[(so + 3) * 6 + r * 2 + 1]);
      break;
    case DM_POLYTOPE_TRIANGLE:
      *rnew = tri_tri[(so + 3) * 6 + r * 2];
      *onew = DMPolytopeTypeComposeOrientation(tct, o, tri_tri[(so + 3) * 6 + r * 2 + 1]);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
  } else if (dim == 3 && sct == DM_POLYTOPE_TETRAHEDRON) {
    switch (tct) {
    case DM_POLYTOPE_POINT:
      break;
    case DM_POLYTOPE_SEGMENT:
      *rnew = tet_seg[(so + 12) * 8 + r * 2];
      *onew = DMPolytopeTypeComposeOrientation(tct, o, tet_seg[(so + 12) * 8 + r * 2 + 1]);
      break;
    case DM_POLYTOPE_TRIANGLE:
      *rnew = tet_tri[(so + 12) * 12 + r * 2];
      *onew = DMPolytopeTypeComposeOrientation(tct, o, tet_tri[(so + 12) * 12 + r * 2 + 1]);
      break;
    case DM_POLYTOPE_TETRAHEDRON:
      *rnew = tet_tet[(so + 12) * 8 + r * 2];
      *onew = DMPolytopeTypeComposeOrientation(tct, o, tet_tet[(so + 12) * 8 + r * 2 + 1]);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
  } else {
    PetscCall(DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCellRefine_Alfeld(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DM       dm;
  PetscInt dim;
  /* Add 1 vertex, 3 edges inside every triangle, making 3 new triangles.
   2
   |\
   |\\
   | |\
   | \ \
   | |  \
   |  \  \
   |   |  \
   2   \   \
   |   |    1
   |   2    \
   |   |    \
   |   /\   \
   |  0  1  |
   | /    \ |
   |/      \|
   0---0----1
  */
  static DMPolytopeType triT[] = {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       triS[] = {1, 3, 3};
  static PetscInt triC[] = {DM_POLYTOPE_POINT, 2, 0, 0, 0, DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_POINT, 2, 1, 0, 0, DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_POINT, 2, 2, 0, 0, DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 0, 1, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0, 2, DM_POLYTOPE_SEGMENT, 0, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 0, 2};
  static PetscInt triO[] = {0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, -1, 0, 0, -1};
  /* Add 6 triangles inside every cell, making 4 new tets
     The vertices of our reference tet are [(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1)], which we call [v0, v1, v2, v3]. The first
     three edges are [v0, v1], [v1, v2], [v2, v0] called e0, e1, and e2, and then three edges to the top point [v0, v3], [v1, v3], [v2, v3]
     called e3, e4, and e5. The faces of a tet, given in DMPlexGetRawFaces_Internal() are
       [v0, v1, v2], [v0, v3, v1], [v0, v2, v3], [v2, v1, v3]
     We make a new tet on each face
       [v0, v1, v2, (c0, 0)]
       [v0, v3, v1, (c0, 0)]
       [v0, v2, v3, (c0, 0)]
       [v2, v1, v3, (c0, 0)]
     We create a new face for each edge
       [v0, (c0, 0), v1     ]
       [v0, v2,      (c0, 0)]
       [v2, v1,      (c0, 0)]
       [v0, (c0, 0), v3     ]
       [v1, v3,      (c0, 0)]
       [v3, v2,      (c0, 0)]
   */
  static DMPolytopeType tetT[] = {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_TETRAHEDRON};
  static PetscInt       tetS[] = {1, 4, 6, 4};
  static PetscInt tetC[] = {DM_POLYTOPE_POINT, 3, 0, 0, 0, 0, DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_POINT, 3, 0, 1, 0, 0, DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_POINT, 3, 0, 2, 0, 0, DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_POINT, 3, 1, 0, 1, 0, DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 0, 1, DM_POLYTOPE_SEGMENT, 2, 0, 0, 0, DM_POLYTOPE_SEGMENT, 2, 0, 2, 0, DM_POLYTOPE_SEGMENT, 0, 2, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 2, 0, 1, 0, DM_POLYTOPE_SEGMENT, 0, 1, DM_POLYTOPE_SEGMENT, 0, 2, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 0, 3, DM_POLYTOPE_SEGMENT, 2, 1, 0, 0, DM_POLYTOPE_SEGMENT, 2, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0, 3, DM_POLYTOPE_SEGMENT, 0, 1, DM_POLYTOPE_SEGMENT, 2, 2, 1, 0, DM_POLYTOPE_SEGMENT, 0, 2, DM_POLYTOPE_SEGMENT, 0, 3, DM_POLYTOPE_TRIANGLE, 1, 0, 0, DM_POLYTOPE_TRIANGLE, 0, 0, DM_POLYTOPE_TRIANGLE, 0, 1, DM_POLYTOPE_TRIANGLE, 0, 2, DM_POLYTOPE_TRIANGLE, 1, 1, 0, DM_POLYTOPE_TRIANGLE, 0, 3, DM_POLYTOPE_TRIANGLE, 0, 0, DM_POLYTOPE_TRIANGLE, 0, 4, DM_POLYTOPE_TRIANGLE, 1, 2, 0, DM_POLYTOPE_TRIANGLE, 0, 1, DM_POLYTOPE_TRIANGLE, 0, 3, DM_POLYTOPE_TRIANGLE, 0, 5, DM_POLYTOPE_TRIANGLE, 1, 3, 0, DM_POLYTOPE_TRIANGLE, 0, 2, DM_POLYTOPE_TRIANGLE, 0, 5, DM_POLYTOPE_TRIANGLE, 0, 4};
  static PetscInt tetO[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, -1, -1, 0, -1, 0, -1, -1, -1, 0, -1, -1, 0, -1, 0, 0, 0, 0, 0, 0, -2, 0, 0, -2, -2, 0, 0, -2, -3, -3};

  PetscFunctionBeginHot;
  if (rt) *rt = 0;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  if (dim == 2 && source == DM_POLYTOPE_TRIANGLE) {
    *Nt     = 3;
    *target = triT;
    *size   = triS;
    *cone   = triC;
    *ornt   = triO;
  } else if (dim == 3 && source == DM_POLYTOPE_TETRAHEDRON) {
    *Nt     = 4;
    *target = tetT;
    *size   = tetS;
    *cone   = tetC;
    *ornt   = tetO;
  } else {
    PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, rt, Nt, target, size, cone, ornt));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformInitialize_Alfeld(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                  = DMPlexTransformView_Alfeld;
  tr->ops->setup                 = DMPlexTransformSetUp_Alfeld;
  tr->ops->destroy               = DMPlexTransformDestroy_Alfeld;
  tr->ops->celltransform         = DMPlexTransformCellRefine_Alfeld;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_Alfeld;
  tr->ops->mapcoordinates        = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Alfeld(DMPlexTransform tr)
{
  DMPlexRefine_Alfeld *f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&f));
  tr->data = f;

  PetscCall(DMPlexTransformInitialize_Alfeld(tr));
  PetscFunctionReturn(0);
}
