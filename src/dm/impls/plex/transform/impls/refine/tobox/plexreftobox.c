#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformView_ToBox(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    const char *name;

    CHKERRQ(PetscObjectGetName((PetscObject) tr, &name));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Transformation to box cells %s\n", name ? name : ""));
  } else {
    SETERRQ(PetscObjectComm((PetscObject) tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject) viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetUp_ToBox(DMPlexTransform tr)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_ToBox(DMPlexTransform tr)
{
  DMPlexRefine_ToBox *f = (DMPlexRefine_ToBox *) tr->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(f));
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_ToBox(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  PetscBool      convertTensor = PETSC_TRUE;
  static PetscInt tri_seg[]  = {0, 0, 2, 0, 1, 0,
                                2, 0, 1, 0, 0, 0,
                                1, 0, 0, 0, 2, 0,
                                0, 0, 1, 0, 2, 0,
                                1, 0, 2, 0, 0, 0,
                                2, 0, 0, 0, 1, 0};
  static PetscInt tri_quad[] = {1, -3, 0, -3, 2, -4,
                                0, -2, 2, -2, 1, -2,
                                2, -1, 1, -4, 0, -1,
                                0,  0, 1,  0, 2,  0,
                                1,  1, 2,  2, 0,  1,
                                2,  3, 0,  3, 1,  2};
  static PetscInt tseg_seg[]  = {0, -1,
                                 0,  0,
                                 0,  0,
                                 0, -1};
  static PetscInt tseg_quad[] = {1,  2, 0,  2,
                                 1, -3, 0, -3,
                                 0,  0, 1,  0,
                                 0, -1, 1, -1};
  static PetscInt tet_seg[]  = {3, 0, 2, 0, 0, 0, 1, 0,
                                3, 0, 1, 0, 2, 0, 0, 0,
                                3, 0, 0, 0, 1, 0, 2, 0,
                                2, 0, 3, 0, 1, 0, 0, 0,
                                2, 0, 0, 0, 3, 0, 1, 0,
                                2, 0, 1, 0, 0, 0, 3, 0,
                                1, 0, 0, 0, 2, 0, 3, 0,
                                1, 0, 3, 0, 0, 0, 2, 0,
                                1, 0, 2, 0, 3, 0, 0, 0,
                                0, 0, 1, 0, 3, 0, 2, 0,
                                0, 0, 2, 0, 1, 0, 3, 0,
                                0, 0, 3, 0, 2, 0, 1, 0,
                                0, 0, 1, 0, 2, 0, 3, 0,
                                0, 0, 3, 0, 1, 0, 2, 0,
                                0, 0, 2, 0, 3, 0, 1, 0,
                                1, 0, 0, 0, 3, 0, 2, 0,
                                1, 0, 2, 0, 0, 0, 3, 0,
                                1, 0, 3, 0, 2, 0, 0, 0,
                                2, 0, 3, 0, 0, 0, 1, 0,
                                2, 0, 1, 0, 3, 0, 0, 0,
                                2, 0, 0, 0, 1, 0, 3, 0,
                                3, 0, 2, 0, 1, 0, 0, 0,
                                3, 0, 0, 0, 2, 0, 1, 0,
                                3, 0, 1, 0, 0, 0, 2, 0};
  static PetscInt tet_quad[] = {2, 0, 5, -3, 4, 0, 0, 3, 3, 1, 1, 1,
                                0, 0, 3, 0, 5, 0, 1, 0, 4, -2, 2, 0,
                                1, 1, 4, -3, 3, -3, 2, 3, 5, -2, 0, 0,
                                3, 1, 5, 3, 0, 0, 4, 3, 2, 0, 1, -3,
                                4, 0, 2, 3, 5, -2, 1, -4, 0, -2, 3, 1,
                                1, -3, 0, -3, 2, -2, 3, 0, 5, 0, 4, 0,
                                2, -2, 1, -4, 0, -2, 4, -3, 3, -3, 5, 0,
                                4, -2, 3, -4, 1, 1, 5, 3, 0, 0, 2, -2,
                                5, 0, 0, 3, 3, 1, 2, -3, 1, -3, 4, -2,
                                3, -3, 1, 0, 4, -2, 0, -3, 2, -2, 5, -2,
                                0, -2, 2, -3, 1, -3, 5, -3, 4, 0, 3, -3,
                                5, -2, 4, 3, 2, 0, 3, -4, 1, 1, 0, -2,
                                0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0,
                                3, 1, 4, 3, 1, -3, 5, 3, 2, -2, 0, 0,
                                5, 0, 2, -3, 4, -2, 0, 3, 1, 1, 3, 1,
                                4, 0, 1, -4, 3, 1, 2, 3, 0, 0, 5, -2,
                                2, 0, 0, 3, 1, 1, 5, -3, 3, -3, 4, 0,
                                5, -2, 3, -4, 0, -2, 4, 3, 1, -3, 2, 0,
                                4, -2, 5, 3, 2, -2, 3, -4, 0, -2, 1, 1,
                                3, -3, 0, -3, 5, -2, 1, 0, 2, 0, 4, -2,
                                1, 1, 2, 3, 0, 0, 4, -3, 5, 0, 3, -3,
                                0, -2, 5, -3, 3, -3, 2, -3, 4, -2, 1, -3,
                                2, -2, 4, -3, 5, 0, 1, -4, 3, 1, 0, -2,
                                1, -3, 3, 0, 4, 0, 0, -3, 5, -2, 2, -2};
  static PetscInt tet_hex[]  = {2, -2, 3, -2, 1, -10, 0, -13,
                                3, -10, 1, -13, 2, -10, 0, -10,
                                1, -2, 2, -13, 3, -13, 0, -2,
                                3, -13, 2, -10, 0, -2, 1, -2,
                                2, -13, 0, -10, 3, -2, 1, -13,
                                0, -13, 3, -10, 2, -2, 1, -10,
                                0, -10, 1, -2, 3, -10, 2, -10,
                                1, -10, 3, -13, 0, -13, 2, -2,
                                3, -2, 0, -2, 1, -13, 2, -13,
                                1, -13, 0, -13, 2, -13, 3, -2,
                                0, -2, 2, -2, 1, -2, 3, -13,
                                2, -10, 1, -10, 0, -10, 3, -10,
                                0, 0, 1, 0, 2, 0, 3, 0,
                                1, 17, 2, 17, 0, 17, 3, 16,
                                2, 16, 0, 16, 1, 16, 3, 17,
                                1, 16, 0, 17, 3, 17, 2, 16,
                                0, 16, 3, 16, 1, 0, 2, 17,
                                3, 0, 1, 17, 0, 0, 2, 0,
                                2, 17, 3, 0, 0, 16, 1, 0,
                                3, 17, 0, 0, 2, 16, 1, 16,
                                0, 17, 2, 0, 3, 16, 1, 17,
                                3, 16, 2, 16, 1, 17, 0, 17,
                                2, 0, 1, 16, 3, 0, 0, 0,
                                1, 0, 3, 17, 2, 17, 0, 16};
  static PetscInt trip_seg[]  = {1, 0, 0, 0, 3, 0, 4, 0, 2, 0,
                                 1, 0, 0, 0, 4, 0, 2, 0, 3, 0,
                                 1, 0, 0, 0, 2, 0, 3, 0, 4, 0,
                                 0, 0, 1, 0, 3, 0, 2, 0, 4, 0,
                                 0, 0, 1, 0, 4, 0, 3, 0, 2, 0,
                                 0, 0, 1, 0, 2, 0, 4, 0, 3, 0,
                                 0, 0, 1, 0, 2, 0, 3, 0, 4, 0,
                                 0, 0, 1, 0, 4, 0, 2, 0, 3, 0,
                                 0, 0, 1, 0, 3, 0, 4, 0, 2, 0,
                                 1, 0, 0, 0, 2, 0, 4, 0, 3, 0,
                                 1, 0, 0, 0, 4, 0, 3, 0, 2, 0,
                                 1, 0, 0, 0, 3, 0, 2, 0, 4, 0};
  static PetscInt trip_quad[] = {1, 1, 2, 2, 0, 1, 7, -1, 8, -1, 6, -1, 4, -1, 5, -1, 3, -1,
                                 2, 3, 0, 3, 1, 2, 8, -1, 6, -1, 7, -1, 5, -1, 3, -1, 4, -1,
                                 0, 0, 1, 0, 2, 0, 6, -1, 7, -1, 8, -1, 3, -1, 4, -1, 5, -1,
                                 2, -1, 1, -4, 0, -1, 4, 0, 3, 0, 5, 0, 7, 0, 6, 0, 8, 0,
                                 0, -2, 2, -2, 1, -2, 5, 0, 4, 0, 3, 0, 8, 0, 7, 0, 6, 0,
                                 1, -3, 0, -3, 2, -4, 3, 0, 5, 0, 4, 0, 6, 0, 8, 0, 7, 0,
                                 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0,
                                 2, 3, 0, 3, 1, 2, 5, 0, 3, 0, 4, 0, 8, 0, 6, 0, 7, 0,
                                 1, 1, 2, 2, 0, 1, 4, 0, 5, 0, 3, 0, 7, 0, 8, 0, 6, 0,
                                 1, -3, 0, -3, 2, -4, 6, -1, 8, -1, 7, -1, 3, -1, 5, -1, 4, -1,
                                 0, -2, 2, -2, 1, -2, 8, -1, 7, -1, 6, -1, 5, -1, 4, -1, 3, -1,
                                 2, -1, 1, -4, 0, -1, 7, -1, 6, -1, 8, -1, 4, -1, 3, -1, 5, -1};
  static PetscInt trip_hex[]  = {4, -12, 5, -6, 3, -12, 1, -12, 2, -6, 0, -12,
                                 5, -11, 3, -11, 4, -6, 2, -11, 0, -11, 1, -6,
                                 3, -9, 4, -9, 5, -9, 0, -9, 1, -9, 2, -9,
                                 2, -3, 1, -4, 0, -3, 5, -3, 4, -4, 3, -3,
                                 0, -2, 2, -2, 1, -2, 3, -2, 5, -2, 4, -2,
                                 1, -1, 0, -1, 2, -4, 4, -1, 3, -1, 5, -4,
                                 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0,
                                 2, 1, 0, 1, 1, 2, 5, 1, 3, 1, 4, 2,
                                 1, 3, 2, 2, 0, 3, 4, 3, 5, 2, 3, 3,
                                 4, 8, 3, 8, 5, 11, 1, 8, 0, 8, 2, 11,
                                 3, 10, 5, 10, 4, 10, 0, 10, 2, 10, 1, 10,
                                 5, 5, 4, 11, 3, 5, 2, 5, 1, 11, 0, 5};
  static PetscInt ctrip_seg[]  = {0, -1,
                                  0, -1,
                                  0, -1,
                                  0, 0,
                                  0, 0,
                                  0, 0,
                                  0, 0,
                                  0, 0,
                                  0, 0,
                                  0, -1,
                                  0, -1,
                                  0, -1};
  static PetscInt ctrip_quad[] = {0, -1, 2, -1, 1, -1,
                                  2, -1, 1, -1, 0, -1,
                                  1, -1, 0, -1, 2, -1,
                                  0, 0, 2, 0, 1, 0,
                                  2, 0, 1, 0, 0, 0,
                                  1, 0, 0, 0, 2, 0,
                                  0, 0, 1, 0, 2, 0,
                                  1, 0, 2, 0, 0, 0,
                                  2, 0, 0, 0, 1, 0,
                                  0, -1, 1, -1, 2, -1,
                                  1, -1, 2, -1, 0, -1,
                                  2, -1, 0, -1, 1, -1};
  static PetscInt ctrip_hex[]  = {1, 8, 0, 8, 2, 11,
                                  0, 10, 2, 10, 1, 10,
                                  2, 5, 1, 11, 0, 5,
                                  1, -1, 0, -1, 2, -4,
                                  0, -2, 2, -2, 1, -2,
                                  2, -3, 1, -4, 0, -3,
                                  0, 0, 1, 0, 2, 0,
                                  1, 3, 2, 2, 0, 3,
                                  2, 1, 0, 1, 1, 2,
                                  0, -9, 1, -9, 2, -9,
                                  1, -12, 2, -6, 0, -12,
                                  2, -11, 0, -11, 1, -6};
  static PetscInt tquadp_seg[]  = {0, -1,
                                   0, -1,
                                   0, -1,
                                   0, -1,
                                   0,  0,
                                   0,  0,
                                   0,  0,
                                   0,  0,
                                   0,  0,
                                   0,  0,
                                   0,  0,
                                   0,  0,
                                   0, -1,
                                   0, -1,
                                   0, -1,
                                   0, -1};
  static PetscInt tquadp_quad[] = {1, -1, 0, -1, 3, -1, 2, -1,
                                   0, -1, 3, -1, 2, -1, 1, -1,
                                   3, -1, 2, -1, 1, -1, 0, -1,
                                   2, -1, 1, -1, 0, -1, 3, -1,
                                   1,  0, 0,  0, 3,  0, 2,  0,
                                   0,  0, 3,  0, 2,  0, 1,  0,
                                   3,  0, 2,  0, 1,  0, 0,  0,
                                   2,  0, 1,  0, 0,  0, 3,  0,
                                   0,  0, 1,  0, 2,  0, 3,  0,
                                   1,  0, 2,  0, 3,  0, 0,  0,
                                   2,  0, 3,  0, 0,  0, 1,  0,
                                   3,  0, 0,  0, 1,  0, 2,  0,
                                   0, -1, 1, -1, 2, -1, 3, -1,
                                   1, -1, 2, -1, 3, -1, 0, -1,
                                   2, -1, 3, -1, 0, -1, 1, -1,
                                   3, -1, 0, -1, 1, -1, 2, -1};
  static PetscInt tquadp_hex[]  = {2, 11,  1,  11, 0,  11, 3,  11,
                                   1,  8,  0,   8, 3,   8, 2,   8,
                                   0, 10,  3,  10, 2,  10, 1,  10,
                                   3,  5,  2,   5, 1,   5, 0,   5,
                                   2, -4,  1,  -4, 0,  -4, 3,  -4,
                                   1, -1,  0,  -1, 3,  -1, 2,  -1,
                                   0, -2,  3,  -2, 2,  -2, 1,  -2,
                                   3, -3,  2,  -3, 1,  -3, 0,  -3,
                                   0,   0, 1,   0, 2,   0, 3,   0,
                                   1,   3, 2,   3, 3,   3, 0,   3,
                                   2,   2, 3,   2, 0,   2, 1,   2,
                                   3,   1, 0,   1, 1,   1, 2,   1,
                                   0,  -9, 1,  -9, 2,  -9, 3,  -9,
                                   1, -12, 2, -12, 3, -12, 0, -12,
                                   2,  -6, 3,  -6, 0,  -6, 1,  -6,
                                   3, -11, 0, -11, 1, -11, 2, -11};

  PetscFunctionBeginHot;
  *rnew = r; *onew = o;
  if (!so) PetscFunctionReturn(0);
  if (convertTensor) {
    switch (sct) {
      case DM_POLYTOPE_POINT:
      case DM_POLYTOPE_SEGMENT:
      case DM_POLYTOPE_POINT_PRISM_TENSOR:
      case DM_POLYTOPE_QUADRILATERAL:
      case DM_POLYTOPE_HEXAHEDRON:
        CHKERRQ(DMPlexTransformGetSubcellOrientation_Regular(tr, sct, sp, so, tct, r, o, rnew, onew));
        break;
      case DM_POLYTOPE_TRIANGLE:
      switch (tct) {
        case DM_POLYTOPE_POINT: break;
        case DM_POLYTOPE_SEGMENT:
          *rnew = tri_seg[(so+3)*6 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, tri_seg[(so+3)*6 + r*2 + 1]);
          break;
        case DM_POLYTOPE_QUADRILATERAL:
          *rnew = tri_quad[(so+3)*6 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, tri_quad[(so+3)*6 + r*2 + 1]);
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
      case DM_POLYTOPE_SEG_PRISM_TENSOR:
      switch (tct) {
        case DM_POLYTOPE_SEGMENT:
        case DM_POLYTOPE_POINT_PRISM_TENSOR:
          *rnew = tseg_seg[(so+2)*2 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, tseg_seg[(so+2)*2 + r*2 + 1]);
          break;
        case DM_POLYTOPE_QUADRILATERAL:
          *rnew = tseg_quad[(so+2)*4 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, tseg_quad[(so+2)*4 + r*2 + 1]);
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
      case DM_POLYTOPE_TETRAHEDRON:
      switch (tct) {
        case DM_POLYTOPE_POINT: break;
        case DM_POLYTOPE_SEGMENT:
          *rnew = tet_seg[(so+12)*8 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, tet_seg[(so+12)*8 + r*2 + 1]);
          break;
        case DM_POLYTOPE_QUADRILATERAL:
          *rnew = tet_quad[(so+12)*12 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, tet_quad[(so+12)*12 + r*2 + 1]);
          break;
        case DM_POLYTOPE_HEXAHEDRON:
          *rnew = tet_hex[(so+12)*8 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, tet_hex[(so+12)*8 + r*2 + 1]);
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
      case DM_POLYTOPE_TRI_PRISM:
      switch (tct) {
        case DM_POLYTOPE_POINT: break;
        case DM_POLYTOPE_SEGMENT:
          *rnew = trip_seg[(so+6)*10 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, trip_seg[(so+6)*10 + r*2 + 1]);
          break;
        case DM_POLYTOPE_QUADRILATERAL:
          *rnew = trip_quad[(so+6)*18 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, trip_quad[(so+6)*18 + r*2 + 1]);
          break;
        case DM_POLYTOPE_HEXAHEDRON:
          *rnew = trip_hex[(so+6)*12 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, trip_hex[(so+6)*12 + r*2 + 1]);
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
      case DM_POLYTOPE_TRI_PRISM_TENSOR:
      switch (tct) {
        case DM_POLYTOPE_SEGMENT:
          *rnew = ctrip_seg[(so+6)*2 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, ctrip_seg[(so+6)*2 + r*2 + 1]);
          break;
        case DM_POLYTOPE_QUADRILATERAL:
          *rnew = ctrip_quad[(so+6)*6 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, ctrip_quad[(so+6)*6 + r*2 + 1]);
          break;
        case DM_POLYTOPE_HEXAHEDRON:
          *rnew = ctrip_hex[(so+6)*6 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, ctrip_hex[(so+6)*6 + r*2 + 1]);
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
      case DM_POLYTOPE_QUAD_PRISM_TENSOR:
      switch (tct) {
        case DM_POLYTOPE_SEGMENT:
          *rnew = tquadp_seg[(so+8)*2 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, tquadp_seg[(so+8)*2 + r*2 + 1]);
          break;
        case DM_POLYTOPE_QUADRILATERAL:
          *rnew = tquadp_quad[(so+8)*8 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, tquadp_quad[(so+8)*8 + r*2 + 1]);
          break;
        case DM_POLYTOPE_HEXAHEDRON:
          *rnew = tquadp_hex[(so+8)*8 + r*2];
          *onew = DMPolytopeTypeComposeOrientation(tct, o, tquadp_hex[(so+8)*8 + r*2 + 1]);
          break;
        default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
      }
      break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell type %s", DMPolytopeTypes[sct]);
    }
  } else {
    switch (sct) {
      case DM_POLYTOPE_POINT:
      case DM_POLYTOPE_SEGMENT:
      case DM_POLYTOPE_POINT_PRISM_TENSOR:
      case DM_POLYTOPE_QUADRILATERAL:
      case DM_POLYTOPE_SEG_PRISM_TENSOR:
      case DM_POLYTOPE_HEXAHEDRON:
      case DM_POLYTOPE_QUAD_PRISM_TENSOR:
        CHKERRQ(DMPlexTransformGetSubcellOrientation_Regular(tr, sct, sp, so, tct, r, o, rnew, onew));
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell type %s", DMPolytopeTypes[sct]);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformCellRefine_ToBox(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  PetscBool      convertTensor = PETSC_TRUE;
  /* Change tensor edges to segments */
  static DMPolytopeType tedgeT[]  = {DM_POLYTOPE_SEGMENT};
  static PetscInt       tedgeS[]  = {1};
  static PetscInt       tedgeC[]  = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0};
  static PetscInt       tedgeO[]  = {                         0,                          0};
  /* Add 1 vertex, 3 edges inside every triangle, making 3 new quadrilaterals.
   2
   |\
   | \
   |  \
   |   \
   0    1
   |     \
   |      \
   2       1
   |\     / \
   | 2   1   \
   |  \ /     \
   1   |       0
   |   0        \
   |   |         \
   |   |          \
   0-0-0-----1-----1
  */
  static DMPolytopeType triT[]    = {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT, DM_POLYTOPE_QUADRILATERAL};
  static PetscInt       triS[]    = {1, 3, 3};
  static PetscInt       triC[]    = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 0,    0,
                                     DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 0,    0,
                                     DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_POINT, 0,    0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 1, 2, 1,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 0,    0,
                                     DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0};
  static PetscInt       triO[]    = {0, 0,
                                     0, 0,
                                     0, 0,
                                     0,  0, -1,  0,
                                     0,  0,  0, -1,
                                     0, -1,  0,  0};
  /* Add 1 edge inside every tensor quad, making 2 new quadrilaterals
     2----2----1----3----3
     |         |         |
     |         |         |
     |         |         |
     4    A    6    B    5
     |         |         |
     |         |         |
     |         |         |
     0----0----0----1----1
  */
  static DMPolytopeType tsegT[]  = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_QUADRILATERAL};
  static PetscInt       tsegS[]  = {1, 2};
  static PetscInt       tsegC[]  = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0,
                                     /* TODO  Fix these */
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 2, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 3, 0, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 0,    0};
  static PetscInt       tsegO[]  = {0, 0,
                                    0, 0, -1, -1,
                                    0, 0, -1, -1};
  /* Add 6 triangles inside every cell, making 4 new hexs
     The vertices of our reference tet are [(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1)], which we call [v0, v1, v2, v3]. The first
     three edges are [v0, v1], [v1, v2], [v2, v0] called e0, e1, and e2, and then three edges to the top point [v0, v3], [v1, v3], [v2, v3]
     called e3, e4, and e5. The faces of a tet, given in DMPlexGetRawFaces_Internal() are
       [v0, v1, v2], [v0, v3, v1], [v0, v2, v3], [v2, v1, v3]
     We make a new hex in each corner
       [v0, (e0, 0), (f0, 0), (e2, 0), (e3, 0), (f2, 0), (c0, 0), (f1, 0)]
       [v1, (e4, 0), (f3, 0), (e1, 0), (e0, 0), (f0, 0), (c0, 0), (f1, 0)]
       [v2, (e1, 0), (f3, 0), (e5, 0), (e2, 0), (f2, 0), (c0, 0), (f0, 0)]
       [v3, (e4, 0), (f1, 0), (e3, 0), (e5, 0), (f2, 0), (c0, 0), (f3, 0)]
     We create a new face for each edge
       [(e3, 0), (f2, 0), (c0, 0), (f1, 0)]
       [(f0, 0), (e0, 0), (f1, 0), (c0, 0)]
       [(e2, 0), (f0, 0), (c0, 0), (f2, 0)]
       [(f3, 0), (e4, 0), (f1, 0), (c0, 0)]
       [(e1, 0), (f3, 0), (c0, 0), (f0, 0)]
       [(e5, 0), (f3, 0), (c0, 0), (f2, 0)]
     I could write a program to generate these from the first hex by acting with the symmetry group to take one subcell into another.
   */
  static DMPolytopeType tetT[]    = {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT, DM_POLYTOPE_QUADRILATERAL, DM_POLYTOPE_HEXAHEDRON};
  static PetscInt       tetS[]    = {1, 4, 6, 4};
  static PetscInt       tetC[]    = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 3, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 2, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 1, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 0,    0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 1, 2, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 3, 1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 0,    3,
                                     DM_POLYTOPE_SEGMENT, 1, 3, 0, DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 0, 1,
                                     DM_POLYTOPE_SEGMENT, 1, 3, 2, DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 1, 2, 1,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_QUADRILATERAL, 1, 2, 0, DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 0,    2, DM_POLYTOPE_QUADRILATERAL, 1, 1, 0,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 3, 1, DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 1, 0, 1, DM_POLYTOPE_QUADRILATERAL, 0,    3, DM_POLYTOPE_QUADRILATERAL, 0,    4, DM_POLYTOPE_QUADRILATERAL, 1, 1, 2,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 3, 0, DM_POLYTOPE_QUADRILATERAL, 0,    2, DM_POLYTOPE_QUADRILATERAL, 1, 2, 1, DM_POLYTOPE_QUADRILATERAL, 0,    4, DM_POLYTOPE_QUADRILATERAL, 0,    5, DM_POLYTOPE_QUADRILATERAL, 1, 0, 2,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 1, 1, DM_POLYTOPE_QUADRILATERAL, 0,    5, DM_POLYTOPE_QUADRILATERAL, 1, 2, 2, DM_POLYTOPE_QUADRILATERAL, 0,    3, DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_QUADRILATERAL, 1, 3, 2};
  static PetscInt       tetO[]    = {0, 0,
                                     0, 0,
                                     0, 0,
                                     0, 0,
                                     0,  0, -1, -1,
                                    -1,  0,  0, -1,
                                     0,  0, -1, -1,
                                    -1,  0,  0, -1,
                                     0,  0, -1, -1,
                                     0,  0, -1, -1,
                                     0,  0,  0,  0,  0,  0,
                                     1, -3,  1,  0,  0,  3,
                                     0, -2,  1, -3,  0,  3,
                                     1, -2,  3, -4, -2,  3};
  /* Add 3 quads inside every triangular prism, making 4 new prisms. */
  static DMPolytopeType tripT[]   = {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT, DM_POLYTOPE_QUADRILATERAL, DM_POLYTOPE_HEXAHEDRON};
  static PetscInt       tripS[]   = {1, 5, 9, 6};
  static PetscInt       tripC[]   = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 3, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 4, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 3, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 0,    4, DM_POLYTOPE_SEGMENT, 1, 4, 1,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 1, 3, 3, DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 0,    2,
                                     DM_POLYTOPE_SEGMENT, 0,    4, DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 1, 3, 1, DM_POLYTOPE_SEGMENT, 1, 4, 3,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 1, 2, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 1, 3, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    4, DM_POLYTOPE_SEGMENT, 1, 4, 0,
                                     DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 2, 2,
                                     DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 3, 2,
                                     DM_POLYTOPE_SEGMENT, 0,    4, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_SEGMENT, 1, 4, 2,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_QUADRILATERAL, 1, 2, 0, DM_POLYTOPE_QUADRILATERAL, 0,    5, DM_POLYTOPE_QUADRILATERAL, 0,    3, DM_POLYTOPE_QUADRILATERAL, 1, 4, 1,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 2, DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 1, 2, 1, DM_POLYTOPE_QUADRILATERAL, 0,    4, DM_POLYTOPE_QUADRILATERAL, 1, 3, 0, DM_POLYTOPE_QUADRILATERAL, 0,    3,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 1, DM_POLYTOPE_QUADRILATERAL, 0,    2, DM_POLYTOPE_QUADRILATERAL, 0,    5, DM_POLYTOPE_QUADRILATERAL, 1, 3, 1, DM_POLYTOPE_QUADRILATERAL, 0,    4, DM_POLYTOPE_QUADRILATERAL, 1, 4, 0,
                                     DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_QUADRILATERAL, 1, 1, 0, DM_POLYTOPE_QUADRILATERAL, 1, 2, 3, DM_POLYTOPE_QUADRILATERAL, 0,    8, DM_POLYTOPE_QUADRILATERAL, 0,    6, DM_POLYTOPE_QUADRILATERAL, 1, 4, 2,
                                     DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 1, 1, 1, DM_POLYTOPE_QUADRILATERAL, 1, 2, 2, DM_POLYTOPE_QUADRILATERAL, 0,    7, DM_POLYTOPE_QUADRILATERAL, 1, 3, 3, DM_POLYTOPE_QUADRILATERAL, 0,    6,
                                     DM_POLYTOPE_QUADRILATERAL, 0,    2, DM_POLYTOPE_QUADRILATERAL, 1, 1, 2, DM_POLYTOPE_QUADRILATERAL, 0,    8, DM_POLYTOPE_QUADRILATERAL, 1, 3, 2, DM_POLYTOPE_QUADRILATERAL, 0,    7, DM_POLYTOPE_QUADRILATERAL, 1, 4, 3};
  static PetscInt       tripO[]   = {0, 0,
                                     0, 0,
                                     0, 0,
                                     0, 0,
                                     0, 0,
                                     0,  0, -1, -1,
                                    -1,  0,  0, -1,
                                     0, -1, -1,  0,
                                     0,  0, -1, -1,
                                     0,  0, -1, -1,
                                     0,  0, -1, -1,
                                     0, -1, -1,  0,
                                     0, -1, -1,  0,
                                     0, -1, -1,  0,
                                     0,  0,  0, -3,  0,  1,
                                     0,  0,  0,  0,  0, -2,
                                     0,  0,  0,  0, -3,  1,
                                    -2,  0,  0, -3,  0,  1,
                                    -2,  0,  0,  0,  0, -2,
                                    -2,  0,  0,  0, -3,  1};
  /* Add 3 tensor quads inside every tensor triangular prism, making 4 new tensor triangular prisms.
      2
      |\
      | \
      |  \
      0---1

      2

      0   1

      2
      |\
      | \
      |  \
      0---1
  */
  static DMPolytopeType ttriT[]  = {DM_POLYTOPE_POINT_PRISM_TENSOR, DM_POLYTOPE_SEG_PRISM_TENSOR, DM_POLYTOPE_QUAD_PRISM_TENSOR};
  static PetscInt       ttriS[]  = {1, 3, 3};
  static PetscInt       ttriC[]  = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 2, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 0, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 3, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 0, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 4, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 0, 0,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_QUADRILATERAL, 1, 1, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 2, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    0, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    2, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 4, 1,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 1, DM_POLYTOPE_QUADRILATERAL, 1, 1, 1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 2, 1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 3, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    1, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    0,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 2, DM_POLYTOPE_QUADRILATERAL, 1, 1, 2, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    2, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 3, 1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 4, 0};
  static PetscInt       ttriO[]  = {0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0,  0, -1, 0,
                                     0, 0, 0,  0,  0, -1,
                                     0, 0, 0, -1,  0, 0};
  /* TODO Add 3 quads inside every tensor triangular prism, making 4 new triangular prisms.
      2
      |\
      | \
      |  \
      0---1

      2

      0   1

      2
      |\
      | \
      |  \
      0---1
  */
  static DMPolytopeType ctripT[]  = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_QUADRILATERAL, DM_POLYTOPE_HEXAHEDRON};
  static PetscInt       ctripS[]  = {1, 3, 3};
  static PetscInt       ctripC[]  = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 2, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 3, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_SEGMENT, 1, 4, 0,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_QUADRILATERAL, 1, 1, 0, DM_POLYTOPE_QUADRILATERAL, 1, 2, 0, DM_POLYTOPE_QUADRILATERAL, 0,    2, DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_QUADRILATERAL, 1, 4, 1,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 1, DM_POLYTOPE_QUADRILATERAL, 1, 1, 1, DM_POLYTOPE_QUADRILATERAL, 1, 2, 1, DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 1,  3, 0, DM_POLYTOPE_QUADRILATERAL, 0,    0,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 2, DM_POLYTOPE_QUADRILATERAL, 1, 1, 2, DM_POLYTOPE_QUADRILATERAL, 0,    2, DM_POLYTOPE_QUADRILATERAL, 1, 3, 1, DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 1, 4, 0};
  static PetscInt       ctripO[]  = {0, 0,
                                     0, 0, -1, -1,
                                     0, 0, -1, -1,
                                     0, 0, -1, -1,
                                    -2, 0, 0, -3,  0,  1,
                                    -2, 0, 0,  0,  0, -2,
                                    -2, 0, 0,  0, -3,  1};
  static DMPolytopeType tquadpT[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_QUADRILATERAL, DM_POLYTOPE_HEXAHEDRON};
  static PetscInt       tquadpS[] = {1, 4, 4};
  static PetscInt       tquadpC[] = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 2, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 3, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_SEGMENT, 1, 4, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 3, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 3, DM_POLYTOPE_SEGMENT, 1, 5, 0,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_QUADRILATERAL, 1, 1, 0, DM_POLYTOPE_QUADRILATERAL, 1, 2, 0, DM_POLYTOPE_QUADRILATERAL, 0,    3, DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_QUADRILATERAL, 1, 5, 1,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 1, DM_POLYTOPE_QUADRILATERAL, 1, 1, 1, DM_POLYTOPE_QUADRILATERAL, 1, 2, 1, DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 1, 3, 0, DM_POLYTOPE_QUADRILATERAL, 0,    0,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 2, DM_POLYTOPE_QUADRILATERAL, 1, 1, 2, DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 1, 4, 0, DM_POLYTOPE_QUADRILATERAL, 1, 3, 1, DM_POLYTOPE_QUADRILATERAL, 0,    2,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 3, DM_POLYTOPE_QUADRILATERAL, 1, 1, 3, DM_POLYTOPE_QUADRILATERAL, 0,    3, DM_POLYTOPE_QUADRILATERAL, 1, 4, 1, DM_POLYTOPE_QUADRILATERAL, 0,    2, DM_POLYTOPE_QUADRILATERAL, 1, 5, 0,
  };
  static PetscInt       tquadpO[] = {0,  0,
                                     0,  0, -1, -1,
                                     0,  0, -1, -1,
                                     0,  0, -1, -1,
                                     0,  0, -1, -1,
                                    -2,  0,  0, -3,  0,  1,
                                    -2,  0,  0,  0,  0, -2,
                                    -2,  0, -3,  0,  0,  1,
                                    -2,  0,  0,  0, -3,  1};

  PetscFunctionBeginHot;
  if (rt) *rt = 0;
  if (convertTensor) {
    switch (source) {
      case DM_POLYTOPE_POINT:
      case DM_POLYTOPE_SEGMENT:
      case DM_POLYTOPE_QUADRILATERAL:
      case DM_POLYTOPE_HEXAHEDRON:
        CHKERRQ(DMPlexTransformCellRefine_Regular(tr, source, p, rt, Nt, target, size, cone, ornt));
        break;
      case DM_POLYTOPE_POINT_PRISM_TENSOR: *Nt = 1; *target = tedgeT;  *size = tedgeS;  *cone = tedgeC;  *ornt = tedgeO;  break;
      case DM_POLYTOPE_SEG_PRISM_TENSOR:   *Nt = 2; *target = tsegT;   *size = tsegS;   *cone = tsegC;   *ornt = tsegO;   break;
      case DM_POLYTOPE_TRI_PRISM_TENSOR:   *Nt = 3; *target = ctripT;  *size = ctripS;  *cone = ctripC;  *ornt = ctripO;  break;
      case DM_POLYTOPE_TRIANGLE:           *Nt = 3; *target = triT;    *size = triS;    *cone = triC;    *ornt = triO;    break;
      case DM_POLYTOPE_TETRAHEDRON:        *Nt = 4; *target = tetT;    *size = tetS;    *cone = tetC;    *ornt = tetO;    break;
      case DM_POLYTOPE_TRI_PRISM:          *Nt = 4; *target = tripT;   *size = tripS;   *cone = tripC;   *ornt = tripO;   break;
      case DM_POLYTOPE_QUAD_PRISM_TENSOR:  *Nt = 3; *target = tquadpT; *size = tquadpS; *cone = tquadpC; *ornt = tquadpO; break;
      case DM_POLYTOPE_PYRAMID:            *Nt = 0; *target = NULL;    *size = NULL;    *cone = NULL;    *ornt = NULL;    break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No refinement strategy for %s", DMPolytopeTypes[source]);
    }
  } else {
    switch (source) {
      case DM_POLYTOPE_POINT:
      case DM_POLYTOPE_POINT_PRISM_TENSOR:
      case DM_POLYTOPE_SEGMENT:
      case DM_POLYTOPE_QUADRILATERAL:
      case DM_POLYTOPE_SEG_PRISM_TENSOR:
      case DM_POLYTOPE_HEXAHEDRON:
      case DM_POLYTOPE_QUAD_PRISM_TENSOR:
        CHKERRQ(DMPlexTransformCellRefine_Regular(tr, source, p, rt, Nt, target, size, cone, ornt));
        break;
      case DM_POLYTOPE_TRIANGLE:           *Nt = 3; *target = triT;    *size = triS;    *cone = triC;    *ornt = triO;    break;
      case DM_POLYTOPE_TETRAHEDRON:        *Nt = 4; *target = tetT;    *size = tetS;    *cone = tetC;    *ornt = tetO;    break;
      case DM_POLYTOPE_TRI_PRISM:          *Nt = 4; *target = tripT;   *size = tripS;   *cone = tripC;   *ornt = tripO;   break;
      case DM_POLYTOPE_TRI_PRISM_TENSOR:   *Nt = 3; *target = ttriT;   *size = ttriS;   *cone = ttriC;   *ornt = ttriO;   break;
      case DM_POLYTOPE_PYRAMID:            *Nt = 0; *target = NULL;    *size = NULL;    *cone = NULL;    *ornt = NULL;    break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No refinement strategy for %s", DMPolytopeTypes[source]);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformInitialize_ToBox(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view    = DMPlexTransformView_ToBox;
  tr->ops->setup   = DMPlexTransformSetUp_ToBox;
  tr->ops->destroy = DMPlexTransformDestroy_ToBox;
  tr->ops->celltransform = DMPlexTransformCellRefine_ToBox;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_ToBox;
  tr->ops->mapcoordinates = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_ToBox(DMPlexTransform tr)
{
  DMPlexRefine_ToBox *f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  CHKERRQ(PetscNewLog(tr, &f));
  tr->data = f;

  CHKERRQ(DMPlexTransformInitialize_ToBox(tr));
  PetscFunctionReturn(0);
}
