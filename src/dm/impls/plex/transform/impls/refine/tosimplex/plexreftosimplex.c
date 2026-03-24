#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

static PetscErrorCode DMPlexTransformView_ToSimplex(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    const char *name;

    PetscCall(PetscObjectGetName((PetscObject)tr, &name));
    PetscCall(PetscViewerASCIIPrintf(viewer, "ToSimplex refinement %s\n", name ? name : ""));
  } else {
    SETERRQ(PetscObjectComm((PetscObject)tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject)viewer)->type_name);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformDestroy_ToSimplex(DMPlexTransform tr)
{
  DMPlexRefine_ToSimplex *f = (DMPlexRefine_ToSimplex *)tr->data;

  PetscFunctionBegin;
  PetscCall(PetscFree(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformGetSubcellOrientation_ToSimplex(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  DM              dm;
  PetscInt        dim;
  PetscBool       reflect;
  PetscInt       *quad_tri;
  static PetscInt quad_seg[] = {
    0, -1, /* o = -4 */
    0, -1, /* o = -3 */
    0, -1, /* o = -2 */
    0, -1, /* o = -1 */
    0, 0,  /* o = 0 */
    0, 0,  /* o = 1 */
    0, 0,  /* o = 2 */
    0, 0,  /* o = 3 */
  };
  // TODO: I don't know how to map the subcells into each other because the
  // symmetry isn't there, this is a total guess
  static PetscInt quad_tri_noreflect[] = {
    0, -4, 1, -4, /* o = -4 */
    0, -3, 1, -3, /* o = -3 */
    1, -2, 0, -2, /* o = -2 */
    1, -1, 0, -1, /* o = -1 */
    0, 0,  1, 0,  /* o = 0 */
    1, 1,  0, 1,  /* o = 1 */
    1, 2,  0, 2,  /* o = 2 */
    0, 3,  1, 3,  /* o = 3 */
  };
  static PetscInt quad_tri_reflect[] = {
    0, -4, 1, -4, /* o = -4 */
    1, -3, 0, -3, /* o = -3 */
    1, -2, 1, -2, /* o = -2 */
    0, -1, 1, -1, /* o = -1 */
    0, 0,  1, 0,  /* o = 0 */
    0, 1,  1, 1,  /* o = 1 */
    1, 2,  0, 2,  /* o = 2 */
    1, 3,  0, 3,  /* o = 3 */
  };

  PetscFunctionBeginHot;
  *rnew = r;
  *onew = o;
  if (!so) PetscFunctionReturn(PETSC_SUCCESS);
  PetscCall(DMPlexRefineToSimplexGetReflect(tr, &reflect));
  if (reflect) quad_tri = quad_tri_reflect;
  else quad_tri = quad_tri_noreflect;
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  if (dim == 2 && sct == DM_POLYTOPE_QUADRILATERAL) {
    switch (tct) {
    case DM_POLYTOPE_POINT:
      break;
    case DM_POLYTOPE_SEGMENT:
      *rnew = quad_seg[(so + 4) * 8 + r * 2];
      *onew = DMPolytopeTypeComposeOrientation(tct, o, quad_seg[(so + 4) * 8 + r * 2 + 1]);
      break;
    case DM_POLYTOPE_TRIANGLE:
      *rnew = quad_tri[(so + 4) * 8 + r * 2];
      *onew = DMPolytopeTypeComposeOrientation(tct, o, quad_tri[(so + 4) * 8 + r * 2 + 1]);
      break;
    default:
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
  } else {
    PetscCall(DMPlexTransformGetSubcellOrientationIdentity(tr, sct, sp, so, tct, r, o, rnew, onew));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformCellRefine_ToSimplex(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  DM        dm;
  PetscInt  dim;
  PetscBool reflect;
  PetscInt *quadC, *quadO;
  /* Add 1 edge inside every quad, making 2 new triangles.
   3---2---2       +-------+      +-------+
   |       |       |      /|      |\      |
   |       |       |  1  / |      | \  1  |
   |       |       |    /  |      |  \    |
   3       1  -->  |   0   |  or  |   0   |
   |       |       |  /    |      |    \  |
   |       |       | /  0  |      |  0  \ |
   |       |       |/      |      |      \|
   0---0---1       +-------+      +-------+
  */
  static DMPolytopeType quadT[]           = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       quadS[]           = {1, 2};
  static PetscInt       quadC_noreflect[] = {/* Cone of edge 0, rising left to right */
                                             DM_POLYTOPE_POINT, 2, 0, 0, 0, DM_POLYTOPE_POINT, 2, 2, 0, 0,
                                             /* Cone of cell 0, anticlockwise from the new edge */
                                             DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0,
                                             /* Cone of cell 1, anticlockwise from the new edge */
                                             DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 1, 3, 0};
  static PetscInt       quadC_reflect[]   = {/* Cone of edge 0, rising right to left */
                                             DM_POLYTOPE_POINT, 2, 1, 0, 0, DM_POLYTOPE_POINT, 2, 3, 0, 0,
                                             /* Cone of cell 0, anticlockwise from the new edge */
                                             DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 3, 0, DM_POLYTOPE_SEGMENT, 1, 0, 0,
                                             /* Cone of cell 1, anticlockwise from the new edge */
                                             DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 2, 0};
  static PetscInt       quadO_noreflect[] = {
    0,  0,    /* Cone of edge 0 */
    -1, 0, 0, /* Cone of cell 0 */
    0,  0, 0, /* Cone of cell 1 */
  };
  static PetscInt quadO_reflect[] = {
    0,  0,    /* Cone of edge 0 */
    0,  0, 0, /* Cone of cell 0 */
    -1, 0, 0, /* Cone of cell 1 */
  };

  PetscFunctionBeginHot;
  if (rt) *rt = 0;
  PetscCall(DMPlexRefineToSimplexGetReflect(tr, &reflect));
  if (reflect) {
    quadC = quadC_reflect;
    quadO = quadO_reflect;
  } else {
    quadC = quadC_noreflect;
    quadO = quadO_noreflect;
  }
  PetscCall(DMPlexTransformGetDM(tr, &dm));
  PetscCall(DMGetDimension(dm, &dim));
  if (dim == 2 && source == DM_POLYTOPE_QUADRILATERAL) {
    *Nt     = 2;
    *target = quadT;
    *size   = quadS;
    *cone   = quadC;
    *ornt   = quadO;
  } else {
    PetscCall(DMPlexTransformCellTransformIdentity(tr, source, p, rt, Nt, target, size, cone, ornt));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformSetFromOptions_ToSimplex(DMPlexTransform tr, PetscOptionItems PetscOptionsObject)
{
  DMPlexRefine_ToSimplex *ts = (DMPlexRefine_ToSimplex *)tr->data;
  PetscBool               reflect, flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject, "DMPlexRefine ToSimplex Options");
  PetscCall(PetscOptionsBool("-dm_plex_transform_tosimplex_reflect", "Reflect the transformation", "", ts->reflect, &reflect, &flg));
  if (flg) PetscCall(DMPlexRefineToSimplexSetReflect(tr, reflect));
  PetscOptionsHeadEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexRefineToSimplexGetReflect - Get the flag to reflect the transform

  Not Collective

  Input Parameter:
. tr - The `DMPlexTransform`

  Output Parameter:
. reflect - Whether to reflect the transform

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexRefineToSimplexSetReflect()`
@*/
PetscErrorCode DMPlexRefineToSimplexGetReflect(DMPlexTransform tr, PetscBool *reflect)
{
  DMPlexRefine_ToSimplex *ts = (DMPlexRefine_ToSimplex *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscAssertPointer(reflect, 2);
  *reflect = ts->reflect;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*@
  DMPlexRefineToSimplexSetReflect - Set the flag to reflect the transform

  Not Collective

  Input Parameters:
+ tr      - The `DMPlexTransform`
- reflect - Whether to reflect the transform

  Level: intermediate

.seealso: `DMPlexTransform`, `DMPlexRefineToSimplexGetReflect()`
@*/
PetscErrorCode DMPlexRefineToSimplexSetReflect(DMPlexTransform tr, PetscBool reflect)
{
  DMPlexRefine_ToSimplex *ts = (DMPlexRefine_ToSimplex *)tr->data;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  ts->reflect = reflect;
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode DMPlexTransformInitialize_ToSimplex(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view                  = DMPlexTransformView_ToSimplex;
  tr->ops->setfromoptions        = DMPlexTransformSetFromOptions_ToSimplex;
  tr->ops->destroy               = DMPlexTransformDestroy_ToSimplex;
  tr->ops->setdimensions         = DMPlexTransformSetDimensions_Internal;
  tr->ops->celltransform         = DMPlexTransformCellRefine_ToSimplex;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_ToSimplex;
  tr->ops->mapcoordinates        = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(PETSC_SUCCESS);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_ToSimplex(DMPlexTransform tr)
{
  DMPlexRefine_ToSimplex *ts;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscCall(PetscNew(&ts));
  tr->redFactor = 1.0;
  tr->data      = ts;
  ts->reflect   = PETSC_FALSE;
  PetscCall(DMPlexTransformInitialize_ToSimplex(tr));
  PetscFunctionReturn(PETSC_SUCCESS);
}
