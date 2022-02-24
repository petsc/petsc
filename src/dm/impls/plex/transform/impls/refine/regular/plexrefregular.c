#include <petsc/private/dmplextransformimpl.h> /*I "petscdmplextransform.h" I*/

/* Regular Refinment of Hybrid Meshes

   We would like to express regular refinement as a small set of rules that can be applied on every point of the Plex
   to automatically generate a refined Plex. In fact, we would like these rules to be general enough to encompass other
   transformations, such as changing from one type of cell to another, as simplex to hex.

   To start, we can create a function that takes an original cell type and returns the number of new cells replacing it
   and the types of the new cells.

   We need the group multiplication table for group actions from the dihedral group for each cell type.

   We need an operator which takes in a cell, and produces a new set of cells with new faces and correct orientations. I think
   we can just write this operator for faces with identity, and then compose the face orientation actions to get the actual
   (face, orient) pairs for each subcell.
*/

/*@
  DMPlexRefineRegularGetAffineFaceTransforms - Gets the affine map from the reference face cell to each face in the given cell

  Input Parameters:
+ cr - The DMPlexCellRefiner object
- ct - The cell type

  Output Parameters:
+ Nf   - The number of faces for this cell type
. v0   - The translation of the first vertex for each face
. J    - The Jacobian for each face (map from original cell to subcell)
. invJ - The inverse Jacobian for each face
- detJ - The determinant of the Jacobian for each face

  Note: The Jacobian and inverse Jacboian will be rectangular, and the inverse is really a generalized inverse.
$         v0 + j x_face = x_cell
$    invj (x_cell - v0) = x_face

  Level: developer

.seealso: DMPlexCellRefinerGetAffineTransforms(), Create()
@*/
PetscErrorCode DMPlexRefineRegularGetAffineFaceTransforms(DMPlexTransform tr, DMPolytopeType ct, PetscInt *Nf, PetscReal *v0[], PetscReal *J[], PetscReal *invJ[], PetscReal *detJ[])
{
  /*
   2
   |\
   | \
   |  \
   |   \
   |    \
   |     \
   |      \
   2       1
   |        \
   |         \
   |          \
   0---0-------1
   v0[Nf][dc]:       3 x 2
   J[Nf][df][dc]:    3 x 1 x 2
   invJ[Nf][dc][df]: 3 x 2 x 1
   detJ[Nf]:         3
   */
  static PetscReal tri_v0[]   = {0.0, -1.0,  0.0, 0.0,  -1.0,  0.0};
  static PetscReal tri_J[]    = {1.0, 0.0,  -1.0, 1.0,   0.0, -1.0};
  static PetscReal tri_invJ[] = {1.0, 0.0,  -0.5, 0.5,   0.0, -1.0};
  static PetscReal tri_detJ[] = {1.0,  1.414213562373095,  1.0};
  /*
   3---------2---------2
   |                   |
   |                   |
   |                   |
   3                   1
   |                   |
   |                   |
   |                   |
   0---------0---------1

   v0[Nf][dc]:       4 x 2
   J[Nf][df][dc]:    4 x 1 x 2
   invJ[Nf][dc][df]: 4 x 2 x 1
   detJ[Nf]:         4
   */
  static PetscReal quad_v0[]   = {0.0, -1.0,  1.0, 0.0,   0.0, 1.0  -1.0,  0.0};
  static PetscReal quad_J[]    = {1.0, 0.0,   0.0, 1.0,  -1.0, 0.0,  0.0, -1.0};
  static PetscReal quad_invJ[] = {1.0, 0.0,   0.0, 1.0,  -1.0, 0.0,  0.0, -1.0};
  static PetscReal quad_detJ[] = {1.0,  1.0,  1.0,  1.0};

  PetscFunctionBegin;
  switch (ct) {
    case DM_POLYTOPE_TRIANGLE:      if (Nf) *Nf = 3; if (v0) *v0 = tri_v0;  if (J) *J = tri_J;  if (invJ) *invJ = tri_invJ;  if (detJ) *detJ = tri_detJ;  break;
    case DM_POLYTOPE_QUADRILATERAL: if (Nf) *Nf = 4; if (v0) *v0 = quad_v0; if (J) *J = quad_J; if (invJ) *invJ = quad_invJ; if (detJ) *detJ = quad_detJ; break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported polytope type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}

/*@
  DMPlexRefineRegularGetAffineTransforms - Gets the affine map from the reference cell to each subcell

  Input Parameters:
+ cr - The DMPlexCellRefiner object
- ct - The cell type

  Output Parameters:
+ Nc   - The number of subcells produced from this cell type
. v0   - The translation of the first vertex for each subcell
. J    - The Jacobian for each subcell (map from reference cell to subcell)
- invJ - The inverse Jacobian for each subcell

  Level: developer

.seealso: DMPlexRefineRegularGetAffineFaceTransforms(), DMPLEXREFINEREGULAR
@*/
PetscErrorCode DMPlexRefineRegularGetAffineTransforms(DMPlexTransform tr, DMPolytopeType ct, PetscInt *Nc, PetscReal *v0[], PetscReal *J[], PetscReal *invJ[])
{
  /*
   2
   |\
   | \
   |  \
   |   \
   | C  \
   |     \
   |      \
   2---1---1
   |\  D  / \
   | 2   0   \
   |A \ /  B  \
   0---0-------1
   */
  static PetscReal tri_v0[]   = {-1.0, -1.0,  0.0, -1.0,  -1.0, 0.0,  0.0, -1.0};
  static PetscReal tri_J[]    = {0.5, 0.0,
                                 0.0, 0.5,

                                 0.5, 0.0,
                                 0.0, 0.5,

                                 0.5, 0.0,
                                 0.0, 0.5,

                                 0.0, -0.5,
                                 0.5,  0.5};
  static PetscReal tri_invJ[] = {2.0, 0.0,
                                 0.0, 2.0,

                                 2.0, 0.0,
                                 0.0, 2.0,

                                 2.0, 0.0,
                                 0.0, 2.0,

                                 2.0,  2.0,
                                -2.0,  0.0};
    /*
     3---------2---------2
     |         |         |
     |    D    2    C    |
     |         |         |
     3----3----0----1----1
     |         |         |
     |    A    0    B    |
     |         |         |
     0---------0---------1
     */
  static PetscReal quad_v0[]   = {-1.0, -1.0,  0.0, -1.0,  0.0, 0.0,  -1.0, 0.0};
  static PetscReal quad_J[]    = {0.5, 0.0,
                                  0.0, 0.5,

                                  0.5, 0.0,
                                  0.0, 0.5,

                                  0.5, 0.0,
                                  0.0, 0.5,

                                  0.5, 0.0,
                                  0.0, 0.5};
  static PetscReal quad_invJ[] = {2.0, 0.0,
                                  0.0, 2.0,

                                  2.0, 0.0,
                                  0.0, 2.0,

                                  2.0, 0.0,
                                  0.0, 2.0,

                                  2.0, 0.0,
                                  0.0, 2.0};
    /*
     Bottom (viewed from top)    Top
     1---------2---------2       7---------2---------6
     |         |         |       |         |         |
     |    B    2    C    |       |    H    2    G    |
     |         |         |       |         |         |
     3----3----0----1----1       3----3----0----1----1
     |         |         |       |         |         |
     |    A    0    D    |       |    E    0    F    |
     |         |         |       |         |         |
     0---------0---------3       4---------0---------5
     */
  static PetscReal hex_v0[]   = {-1.0, -1.0, -1.0,  -1.0,  0.0, -1.0,  0.0, 0.0, -1.0,   0.0, -1.0, -1.0,
                                 -1.0, -1.0,  0.0,   0.0, -1.0,  0.0,  0.0, 0.0,  0.0,  -1.0,  0.0,  0.0};
  static PetscReal hex_J[]    = {0.5, 0.0, 0.0,
                                 0.0, 0.5, 0.0,
                                 0.0, 0.0, 0.5,

                                 0.5, 0.0, 0.0,
                                 0.0, 0.5, 0.0,
                                 0.0, 0.0, 0.5,

                                 0.5, 0.0, 0.0,
                                 0.0, 0.5, 0.0,
                                 0.0, 0.0, 0.5,

                                 0.5, 0.0, 0.0,
                                 0.0, 0.5, 0.0,
                                 0.0, 0.0, 0.5,

                                 0.5, 0.0, 0.0,
                                 0.0, 0.5, 0.0,
                                 0.0, 0.0, 0.5,

                                 0.5, 0.0, 0.0,
                                 0.0, 0.5, 0.0,
                                 0.0, 0.0, 0.5,

                                 0.5, 0.0, 0.0,
                                 0.0, 0.5, 0.0,
                                 0.0, 0.0, 0.5,

                                 0.5, 0.0, 0.0,
                                 0.0, 0.5, 0.0,
                                 0.0, 0.0, 0.5};
  static PetscReal hex_invJ[] = {2.0, 0.0, 0.0,
                                 0.0, 2.0, 0.0,
                                 0.0, 0.0, 2.0,

                                 2.0, 0.0, 0.0,
                                 0.0, 2.0, 0.0,
                                 0.0, 0.0, 2.0,

                                 2.0, 0.0, 0.0,
                                 0.0, 2.0, 0.0,
                                 0.0, 0.0, 2.0,

                                 2.0, 0.0, 0.0,
                                 0.0, 2.0, 0.0,
                                 0.0, 0.0, 2.0,

                                 2.0, 0.0, 0.0,
                                 0.0, 2.0, 0.0,
                                 0.0, 0.0, 2.0,

                                 2.0, 0.0, 0.0,
                                 0.0, 2.0, 0.0,
                                 0.0, 0.0, 2.0,

                                 2.0, 0.0, 0.0,
                                 0.0, 2.0, 0.0,
                                 0.0, 0.0, 2.0,

                                 2.0, 0.0, 0.0,
                                 0.0, 2.0, 0.0,
                                 0.0, 0.0, 2.0};

  PetscFunctionBegin;
  switch (ct) {
    case DM_POLYTOPE_TRIANGLE:      if (Nc) *Nc = 4; if (v0) *v0 = tri_v0;  if (J) *J = tri_J;  if (invJ) *invJ = tri_invJ;  break;
    case DM_POLYTOPE_QUADRILATERAL: if (Nc) *Nc = 4; if (v0) *v0 = quad_v0; if (J) *J = quad_J; if (invJ) *invJ = quad_invJ; break;
    case DM_POLYTOPE_HEXAHEDRON:    if (Nc) *Nc = 8; if (v0) *v0 = hex_v0;  if (J) *J = hex_J;  if (invJ) *invJ = hex_invJ;  break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Unsupported polytope type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}

#if 0
static PetscErrorCode DMPlexCellRefinerGetCellVertices_Regular(DMPlexCellRefiner cr, DMPolytopeType ct, PetscInt *Nv, PetscReal *subcellV[])
{
  static PetscReal seg_v[]  = {-1.0,  0.0,  1.0};
  static PetscReal tri_v[]  = {-1.0, -1.0,  1.0, -1.0,  -1.0, 1.0,  0.0, -1.0,  0.0, 0.0,  -1.0, 0.0};
  static PetscReal quad_v[] = {-1.0, -1.0,  1.0, -1.0,   1.0, 1.0,  -1.0, 1.0,  0.0, -1.0,  1.0, 0.0,   0.0, 1.0,  -1.0, 0.0,  0.0, 0.0};
  static PetscReal tet_v[]  = {-1.0, -1.0, -1.0,   0.0, -1.0, -1.0,   1.0, -1.0, -1.0,
                               -1.0,  0.0, -1.0,   0.0,  0.0, -1.0,  -1.0,  1.0, -1.0,
                               -1.0, -1.0,  0.0,   0.0, -1.0,  0.0,  -1.0,  0.0,  0.0,  -1.0, -1.0,  1.0};
  static PetscReal hex_v[]  = {-1.0, -1.0, -1.0,   0.0, -1.0, -1.0,   1.0, -1.0, -1.0,
                               -1.0,  0.0, -1.0,   0.0,  0.0, -1.0,   1.0,  0.0, -1.0,
                               -1.0,  1.0, -1.0,   0.0,  1.0, -1.0,   1.0,  1.0, -1.0,
                               -1.0, -1.0,  0.0,   0.0, -1.0,  0.0,   1.0, -1.0,  0.0,
                               -1.0,  0.0,  0.0,   0.0,  0.0,  0.0,   1.0,  0.0,  0.0,
                               -1.0,  1.0,  0.0,   0.0,  1.0,  0.0,   1.0,  1.0,  0.0,
                               -1.0, -1.0,  1.0,   0.0, -1.0,  1.0,   1.0, -1.0,  1.0,
                               -1.0,  0.0,  1.0,   0.0,  0.0,  1.0,   1.0,  0.0,  1.0,
                               -1.0,  1.0,  1.0,   0.0,  1.0,  1.0,   1.0,  1.0,  1.0};

  PetscFunctionBegin;
  switch (ct) {
    case DM_POLYTOPE_SEGMENT:       *Nv =  3; *subcellV = seg_v;  break;
    case DM_POLYTOPE_TRIANGLE:      *Nv =  6; *subcellV = tri_v;  break;
    case DM_POLYTOPE_QUADRILATERAL: *Nv =  9; *subcellV = quad_v; break;
    case DM_POLYTOPE_TETRAHEDRON:   *Nv = 10; *subcellV = tet_v;  break;
    case DM_POLYTOPE_HEXAHEDRON:    *Nv = 27; *subcellV = hex_v;  break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No subcell vertices for cell type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexCellRefinerGetSubcellVertices_Regular(DMPlexCellRefiner cr, DMPolytopeType ct, DMPolytopeType rct, PetscInt r, PetscInt *Nv, PetscInt *subcellV[])
{
  static PetscInt seg_v[]  = {0, 1, 1, 2};
  static PetscInt tri_v[]  = {0, 3, 5,  3, 1, 4,  5, 4, 2,  3, 4, 5};
  static PetscInt quad_v[] = {0, 4, 8, 7,  4, 1, 5, 8,  8, 5, 2, 6,  7, 8, 6, 3};
  static PetscInt tet_v[]  = {0, 3, 1, 6,  3, 2, 4, 8,  1, 4, 5, 7,  6, 8, 7, 9,
                              1, 6, 3, 7,  8, 4, 3, 7,  7, 3, 1, 4,  7, 3, 8, 6};
  static PetscInt hex_v[]  = {0,  3,  4,  1,  9, 10, 13, 12,   3,  6,  7,  4, 12, 13, 16, 15,   4,  7,  8,  5, 13, 14, 17, 16,   1,  4 , 5 , 2, 10, 11, 14, 13,
                              9, 12, 13, 10, 18, 19, 22, 21,  10, 13, 14, 11, 19, 20, 23, 22,  13, 16, 17, 14, 22, 23, 26, 25,  12, 15, 16, 13, 21, 22, 25, 24};

  PetscFunctionBegin;
  PetscCheckFalse(ct != rct,PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Cell type %s does not produce %s", DMPolytopeTypes[ct], DMPolytopeTypes[rct]);
  switch (ct) {
    case DM_POLYTOPE_SEGMENT:       *Nv = 2; *subcellV = &seg_v[r*(*Nv)];  break;
    case DM_POLYTOPE_TRIANGLE:      *Nv = 3; *subcellV = &tri_v[r*(*Nv)];  break;
    case DM_POLYTOPE_QUADRILATERAL: *Nv = 4; *subcellV = &quad_v[r*(*Nv)]; break;
    case DM_POLYTOPE_TETRAHEDRON:   *Nv = 4; *subcellV = &tet_v[r*(*Nv)];  break;
    case DM_POLYTOPE_HEXAHEDRON:    *Nv = 8; *subcellV = &hex_v[r*(*Nv)];  break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "No subcell vertices for cell type %s", DMPolytopeTypes[ct]);
  }
  PetscFunctionReturn(0);
}
#endif

static PetscErrorCode DMPlexTransformView_Regular(DMPlexTransform tr, PetscViewer viewer)
{
  PetscBool      isascii;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  PetscValidHeaderSpecific(viewer, PETSC_VIEWER_CLASSID, 2);
  CHKERRQ(PetscObjectTypeCompare((PetscObject) viewer, PETSCVIEWERASCII, &isascii));
  if (isascii) {
    const char *name;

    CHKERRQ(PetscObjectGetName((PetscObject) tr, &name));
    CHKERRQ(PetscViewerASCIIPrintf(viewer, "Regular refinement %s\n", name ? name : ""));
  } else {
    SETERRQ(PetscObjectComm((PetscObject) tr), PETSC_ERR_SUP, "Viewer type %s not yet supported for DMPlexTransform writing", ((PetscObject) viewer)->type_name);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformSetUp_Regular(DMPlexTransform tr)
{
  PetscFunctionBegin;
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformDestroy_Regular(DMPlexTransform tr)
{
  DMPlexRefine_Regular *f = (DMPlexRefine_Regular *) tr->data;

  PetscFunctionBegin;
  CHKERRQ(PetscFree(f));
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformGetSubcellOrientation_Regular(DMPlexTransform tr, DMPolytopeType sct, PetscInt sp, PetscInt so, DMPolytopeType tct, PetscInt r, PetscInt o, PetscInt *rnew, PetscInt *onew)
{
  static PetscInt seg_seg[] = {1, -1, 0, -1,
                               0,  0, 1,  0};
  static PetscInt tri_seg[] = {2, -1, 1, -1, 0, -1,
                               1, -1, 0, -1, 2, -1,
                               0, -1, 2, -1, 1, -1,
                               0,  0, 1,  0, 2,  0,
                               1,  0, 2,  0, 0,  0,
                               2,  0, 0,  0, 1,  0};
  static PetscInt tri_tri[] = {1, -3, 0, -3, 2, -3, 3, -2,
                               0, -2, 2, -2, 1, -2, 3, -1,
                               2, -1, 1, -1, 0, -1, 3, -3,
                               0,  0, 1,  0, 2,  0, 3,  0,
                               1,  1, 2,  1, 0,  1, 3,  1,
                               2,  2, 0,  2, 1,  2, 3,  2};
  static PetscInt quad_seg[]  = {1, 0, 0, 0, 3, 0, 2, 0,
                                 0, 0, 3, 0, 2, 0, 1, 0,
                                 3, 0, 2, 0, 1, 0, 0, 0,
                                 2, 0, 1, 0, 0, 0, 3, 0,
                                 0, 0, 1, 0, 2, 0, 3, 0,
                                 1, 0, 2, 0, 3, 0, 0, 0,
                                 2, 0, 3, 0, 0, 0, 1, 0,
                                 3, 0, 0, 0, 1, 0, 2, 0};
  static PetscInt quad_quad[] = {2, -4, 1, -4, 0, -4, 3, -4,
                                 1, -3, 0, -3, 3, -3, 2, -3,
                                 0, -2, 3, -2, 2, -2, 1, -2,
                                 3, -1, 2, -1, 1, -1, 0, -1,
                                 0,  0, 1,  0, 2,  0, 3,  0,
                                 1,  1, 2,  1, 3,  1, 0,  1,
                                 2,  2, 3,  2, 0,  2, 1,  2,
                                 3,  3, 0,  3, 1,  3, 2,  3};
  static PetscInt tseg_seg[]  = {0, -1,
                                 0,  0,
                                 0,  0,
                                 0, -1};
  static PetscInt tseg_tseg[] = {1, -2, 0, -2,
                                 1, -1, 0, -1,
                                 0,  0, 1,  0,
                                 0,  1, 1,  1};
  static PetscInt tet_seg[]   = {0, -1,
                                 0,  0,
                                 0,  0,
                                 0, -1,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0,  0,
                                 0, -1,
                                 0,  0,
                                 0,  0,
                                 0, -1,
                                 0,  0,
                                 0,  0};
  static PetscInt tet_tri[]   = {2, -1, 3, -1, 1, -3, 0, -2, 6, 1, 7, -3, 5, 2, 4, -3,
                                 3, -1, 1, -1, 2, -3, 0, -1, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, -1, 2, -1, 3, -3, 0, -3, 4, 0, 5, 0, 6, 0, 7, 0,
                                 3, -2, 2, -3, 0, -1, 1, -1, 7, -3, 6, 1, 4, 2, 5, -3,
                                 2, -3, 0, -2, 3, -1, 1, -3, 4, 0, 5, 0, 6, 0, 7, 0,
                                 0, -2, 3, -2, 2, -2, 1, -2, 4, 0, 5, 0, 6, 0, 7, 0,
                                 0, -1, 1, -2, 3, -2, 2, -2, 7, 1, 6, -3, 5, -3, 4, 2,
                                 1, -2, 3, -3, 0, -3, 2, -1, 4, 0, 5, 0, 6, 0, 7, 0,
                                 3, -3, 0, -1, 1, -1, 2, -3, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, -3, 0, -3, 2, -1, 3, -3, 6, -3, 7, 1, 4, -3, 5, 2,
                                 0, -3, 2, -2, 1, -2, 3, -2, 4, 0, 5, 0, 6, 0, 7, 0,
                                 2, -2, 1, -3, 0, -2, 3, -1, 4, 0, 5, 0, 6, 0, 7, 0,
                                 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, 0, 2, 2, 0, 1, 3, 1, 4, 0, 5, 0, 6, 0, 7, 0,
                                 2, 2, 0, 0, 1, 1, 3, 2, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, 2, 0, 1, 3, 1, 2, 2, 5, 0, 4, 0, 7, -1, 6, -1,
                                 0, 1, 3, 0, 1, 0, 2, 0, 4, 0, 5, 0, 6, 0, 7, 0,
                                 3, 0, 1, 2, 0, 2, 2, 1, 4, 0, 5, 0, 6, 0, 7, 0,
                                 2, 0, 3, 2, 0, 0, 1, 1, 4, -2, 5, -2, 7, 0, 6, 0,
                                 3, 2, 0, 2, 2, 1, 1, 2, 4, 0, 5, 0, 6, 0, 7, 0,
                                 0, 2, 2, 0, 3, 0, 1, 0, 4, 0, 5, 0, 6, 0, 7, 0,
                                 3, 1, 2, 1, 1, 2, 0, 2, 5, -2, 4, -2, 6, -1, 7, -1,
                                 2, 1, 1, 1, 3, 2, 0, 0, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, 1, 3, 1, 2, 2, 0, 1, 4, 0, 5, 0, 6, 0, 7, 0};
  static PetscInt tet_tet[]   = {2, -12, 3, -12, 1, -12, 0, -12, 6, -9, 7, -9, 5, -12, 4, -12,
                                 3, -11, 1, -11, 2, -11, 0, -11, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, -10, 2, -10, 3, -10, 0, -10, 4, 0, 5, 0, 6, 0, 7, 0,
                                 3, -9, 2, -9, 0, -9, 1, -9, 7, -9, 6, -9, 4, -12, 5, -12,
                                 2, -8, 0, -8, 3, -8, 1, -8, 4, 0, 5, 0, 6, 0, 7, 0,
                                 0, -7, 3, -7, 2, -7, 1, -7, 4, 0, 5, 0, 6, 0, 7, 0,
                                 0, -6, 1, -6, 3, -6, 2, -6, 4, -3, 5, -3, 7, -6, 6, -6,
                                 1, -5, 3, -5, 0, -5, 2, -5, 4, 0, 5, 0, 6, 0, 7, 0,
                                 3, -4, 0, -4, 1, -4, 2, -4, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, -3, 0, -3, 2, -3, 3, -3, 5, -3, 4, -3, 6, -6, 7, -6,
                                 0, -2, 2, -2, 1, -2, 3, -2, 4, 0, 5, 0, 6, 0, 7, 0,
                                 2, -1, 1, -1, 0, -1, 3, -1, 4, 0, 5, 0, 6, 0, 7, 0,
                                 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, 1, 2, 1, 0, 1, 3, 1, 4, 0, 5, 0, 6, 0, 7, 0,
                                 2, 2, 0, 2, 1, 2, 3, 2, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, 3, 0, 3, 3, 3, 2, 3, 5, 0, 4, 0, 7, 0, 6, 0,
                                 0, 4, 3, 4, 1, 4, 2, 4, 4, 0, 5, 0, 6, 0, 7, 0,
                                 3, 5, 1, 5, 0, 5, 2, 5, 4, 0, 5, 0, 6, 0, 7, 0,
                                 2, 6, 3, 6, 0, 6, 1, 6, 6, 6, 7, 6, 4, 6, 5, 6,
                                 3, 7, 0, 7, 2, 7, 1, 7, 4, 0, 5, 0, 6, 0, 7, 0,
                                 0, 8, 2, 8, 3, 8, 1, 8, 4, 0, 5, 0, 6, 0, 7, 0,
                                 3, 9, 2, 9, 1, 9, 0, 9, 7, 6, 6, 6, 5, 6, 4, 6,
                                 2, 10, 1, 10, 3, 10, 0, 10, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, 11, 3, 11, 2, 11, 0, 11, 4, 0, 5, 0, 6, 0, 7, 0};
  static PetscInt hex_seg[]   = {2, 0, 3, 0, 4, 0, 5, 0, 1, 0, 0, 0,
                                 4, 0, 5, 0, 0, 0, 1, 0, 3, 0, 2, 0,
                                 5, 0, 4, 0, 1, 0, 0, 0, 3, 0, 2, 0,
                                 3, 0, 2, 0, 4, 0, 5, 0, 0, 0, 1, 0,
                                 3, 0, 2, 0, 5, 0, 4, 0, 1, 0, 0, 0,
                                 4, 0, 5, 0, 1, 0, 0, 0, 2, 0, 3, 0,
                                 2, 0, 3, 0, 5, 0, 4, 0, 0, 0, 1, 0,
                                 5, 0, 4, 0, 0, 0, 1, 0, 2, 0, 3, 0,
                                 4, 0, 5, 0, 3, 0, 2, 0, 1, 0, 0, 0,
                                 5, 0, 4, 0, 3, 0, 2, 0, 0, 0, 1, 0,
                                 3, 0, 2, 0, 1, 0, 0, 0, 4, 0, 5, 0,
                                 2, 0, 3, 0, 0, 0, 1, 0, 4, 0, 5, 0,
                                 1, 0, 0, 0, 4, 0, 5, 0, 3, 0, 2, 0,
                                 1, 0, 0, 0, 5, 0, 4, 0, 2, 0, 3, 0,
                                 5, 0, 4, 0, 2, 0, 3, 0, 1, 0, 0, 0,
                                 1, 0, 0, 0, 2, 0, 3, 0, 4, 0, 5, 0,
                                 4, 0, 5, 0, 2, 0, 3, 0, 0, 0, 1, 0,
                                 3, 0, 2, 0, 0, 0, 1, 0, 5, 0, 4, 0,
                                 1, 0, 0, 0, 3, 0, 2, 0, 5, 0, 4, 0,
                                 2, 0, 3, 0, 1, 0, 0, 0, 5, 0, 4, 0,
                                 0, 0, 1, 0, 4, 0, 5, 0, 2, 0, 3, 0,
                                 0, 0, 1, 0, 3, 0, 2, 0, 4, 0, 5, 0,
                                 0, 0, 1, 0, 5, 0, 4, 0, 3, 0, 2, 0,
                                 0, 0, 1, 0, 2, 0, 3, 0, 5, 0, 4, 0,
                                 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0,
                                 0, 0, 1, 0, 5, 0, 4, 0, 2, 0, 3, 0,
                                 0, 0, 1, 0, 3, 0, 2, 0, 5, 0, 4, 0,
                                 0, 0, 1, 0, 4, 0, 5, 0, 3, 0, 2, 0,
                                 2, 0, 3, 0, 1, 0, 0, 0, 4, 0, 5, 0,
                                 1, 0, 0, 0, 3, 0, 2, 0, 4, 0, 5, 0,
                                 3, 0, 2, 0, 0, 0, 1, 0, 4, 0, 5, 0,
                                 4, 0, 5, 0, 2, 0, 3, 0, 1, 0, 0, 0,
                                 1, 0, 0, 0, 2, 0, 3, 0, 5, 0, 4, 0,
                                 5, 0, 4, 0, 2, 0, 3, 0, 0, 0, 1, 0,
                                 1, 0, 0, 0, 5, 0, 4, 0, 3, 0, 2, 0,
                                 1, 0, 0, 0, 4, 0, 5, 0, 2, 0, 3, 0,
                                 2, 0, 3, 0, 0, 0, 1, 0, 5, 0, 4, 0,
                                 3, 0, 2, 0, 1, 0, 0, 0, 5, 0, 4, 0,
                                 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0, 0,
                                 4, 0, 5, 0, 3, 0, 2, 0, 0, 0, 1, 0,
                                 5, 0, 4, 0, 0, 0, 1, 0, 3, 0, 2, 0,
                                 2, 0, 3, 0, 5, 0, 4, 0, 1, 0, 0, 0,
                                 4, 0, 5, 0, 1, 0, 0, 0, 3, 0, 2, 0,
                                 3, 0, 2, 0, 5, 0, 4, 0, 0, 0, 1, 0,
                                 3, 0, 2, 0, 4, 0, 5, 0, 1, 0, 0, 0,
                                 5, 0, 4, 0, 1, 0, 0, 0, 2, 0, 3, 0,
                                 4, 0, 5, 0, 0, 0, 1, 0, 2, 0, 3, 0,
                                 2, 0, 3, 0, 4, 0, 5, 0, 0, 0, 1, 0};
  static PetscInt hex_quad[]   = {7, -2, 4, -2, 5, -2, 6, -2, 8, -3, 11, -3, 10, -3, 9, -3, 3, 1, 2, 1, 1, 1, 0, 1,
                                 8, -2, 9, -2, 10, -2, 11, -2, 3, -4, 0, -4, 1, -4, 2, -4, 7, 0, 4, 0, 5, 0, 6, 0,
                                 9, 1, 8, 1, 11, 1, 10, 1, 0, 3, 3, 3, 2, 3, 1, 3, 5, 2, 6, 2, 7, 2, 4, 2,
                                 6, 3, 5, 3, 4, 3, 7, 3, 10, -1, 9, -1, 8, -1, 11, -1, 2, -4, 3, -4, 0, -4, 1, -4,
                                 4, 1, 7, 1, 6, 1, 5, 1, 11, 2, 8, 2, 9, 2, 10, 2, 1, 3, 0, 3, 3, 3, 2, 3,
                                 10, -4, 11, -4, 8, -4, 9, -4, 2, 1, 1, 1, 0, 1, 3, 1, 6, -1, 5, -1, 4, -1, 7, -1,
                                 5, -4, 6, -4, 7, -4, 4, -4, 9, 0, 10, 0, 11, 0, 8, 0, 0, -2, 1, -2, 2, -2, 3, -2,
                                 11, 3, 10, 3, 9, 3, 8, 3, 1, -2, 2, -2, 3, -2, 0, -2, 4, -3, 7, -3, 6, -3, 5, -3,
                                 11, -1, 8, -1, 9, -1, 10, -1, 7, 3, 4, 3, 5, 3, 6, 3, 2, 2, 1, 2, 0, 2, 3, 2,
                                 10, 2, 9, 2, 8, 2, 11, 2, 5, 1, 6, 1, 7, 1, 4, 1, 1, -1, 2, -1, 3, -1, 0, -1,
                                 5, 2, 4, 2, 7, 2, 6, 2, 1, 2, 0, 2, 3, 2, 2, 2, 10, -4, 9, -4, 8, -4, 11, -4,
                                 4, -3, 5, -3, 6, -3, 7, -3, 0, -3, 1, -3, 2, -3, 3, -3, 8, -2, 11, -2, 10, -2, 9, -2,
                                 3, 1, 0, 1, 1, 1, 2, 1, 9, -4, 8, -4, 11, -4, 10, -4, 6, 3, 7, 3, 4, 3, 5, 3,
                                 1, 3, 2, 3, 3, 3, 0, 3, 10, 1, 11, 1, 8, 1, 9, 1, 5, -4, 4, -4, 7, -4, 6, -4,
                                 8, 0, 11, 0, 10, 0, 9, 0, 4, -4, 7, -4, 6, -4, 5, -4, 0, 0, 3, 0, 2, 0, 1, 0,
                                 0, 0, 1, 0, 2, 0, 3, 0, 5, -1, 4, -1, 7, -1, 6, -1, 9, -3, 8, -3, 11, -3, 10, -3,
                                 9, -3, 10, -3, 11, -3, 8, -3, 6, -2, 5, -2, 4, -2, 7, -2, 3, -3, 0, -3, 1, -3, 2, -3,
                                 7, 0, 6, 0, 5, 0, 4, 0, 2, -1, 3, -1, 0, -1, 1, -1, 11, 3, 8, 3, 9, 3, 10, 3,
                                 2, 2, 3, 2, 0, 2, 1, 2, 6, 2, 7, 2, 4, 2, 5, 2, 10, 2, 11, 2, 8, 2, 9, 2,
                                 6, -1, 7, -1, 4, -1, 5, -1, 3, 0, 2, 0, 1, 0, 0, 0, 9, 1, 10, 1, 11, 1, 8, 1,
                                 2, -4, 1, -4, 0, -4, 3, -4, 11, -2, 10, -2, 9, -2, 8, -2, 7, -2, 6, -2, 5, -2, 4, -2,
                                 1, -1, 0, -1, 3, -1, 2, -1, 4, 0, 5, 0, 6, 0, 7, 0, 11, -1, 10, -1, 9, -1, 8, -1,
                                 0, -2, 3, -2, 2, -2, 1, -2, 8, 3, 9, 3, 10, 3, 11, 3, 4, 1, 5, 1, 6, 1, 7, 1,
                                 3, -3, 2, -3, 1, -3, 0, -3, 7, -3, 6, -3, 5, -3, 4, -3, 8, 0, 9, 0, 10, 0, 11, 0,
                                 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0, 9, 0, 10, 0, 11, 0,
                                 1, 3, 2, 3, 3, 3, 0, 3, 11, -2, 10, -2, 9, -2, 8, -2, 4, 1, 5, 1, 6, 1, 7, 1,
                                 2, 2, 3, 2, 0, 2, 1, 2, 7, -3, 6, -3, 5, -3, 4, -3, 11, -1, 10, -1, 9, -1, 8, -1,
                                 3, 1, 0, 1, 1, 1, 2, 1, 8, 3, 9, 3, 10, 3, 11, 3, 7, -2, 6, -2, 5, -2, 4, -2,
                                 5, 2, 4, 2, 7, 2, 6, 2, 0, -3, 1, -3, 2, -3, 3, -3, 9, 1, 10, 1, 11, 1, 8, 1,
                                 1, -1, 0, -1, 3, -1, 2, -1, 5, -1, 4, -1, 7, -1, 6, -1, 10, 2, 11, 2, 8, 2, 9, 2,
                                 4, -3, 5, -3, 6, -3, 7, -3, 1, 2, 0, 2, 3, 2, 2, 2, 11, 3, 8, 3, 9, 3, 10, 3,
                                 8, 0, 11, 0, 10, 0, 9, 0, 7, 3, 4, 3, 5, 3, 6, 3, 3, -3, 0, -3, 1, -3, 2, -3,
                                 3, -3, 2, -3, 1, -3, 0, -3, 6, 2, 7, 2, 4, 2, 5, 2, 9, -3, 8, -3, 11, -3, 10, -3,
                                 9, -3, 10, -3, 11, -3, 8, -3, 5, 1, 6, 1, 7, 1, 4, 1, 0, 0, 3, 0, 2, 0, 1, 0,
                                 0, -2, 3, -2, 2, -2, 1, -2, 9, -4, 8, -4, 11, -4, 10, -4, 5, -4, 4, -4, 7, -4, 6, -4,
                                 2, -4, 1, -4, 0, -4, 3, -4, 10, 1, 11, 1, 8, 1, 9, 1, 6, 3, 7, 3, 4, 3, 5, 3,
                                 7, 0, 6, 0, 5, 0, 4, 0, 3, 0, 2, 0, 1, 0, 0, 0, 8, -2, 11, -2, 10, -2, 9, -2,
                                 6, -1, 7, -1, 4, -1, 5, -1, 2, -1, 3, -1, 0, -1, 1, -1, 10, -4, 9, -4, 8, -4, 11, -4,
                                 11, -1, 8, -1, 9, -1, 10, -1, 4, -4, 7, -4, 6, -4, 5, -4, 1, -1, 2, -1, 3, -1, 0, -1,
                                 10, 2, 9, 2, 8, 2, 11, 2, 6, -2, 5, -2, 4, -2, 7, -2, 2, 2, 1, 2, 0, 2, 3, 2,
                                 8, -2, 9, -2, 10, -2, 11, -2, 0, 3, 3, 3, 2, 3, 1, 3, 4, -3, 7, -3, 6, -3, 5, -3,
                                 4, 1, 7, 1, 6, 1, 5, 1, 8, -3, 11, -3, 10, -3, 9, -3, 0, -2, 1, -2, 2, -2, 3, -2,
                                 9, 1, 8, 1, 11, 1, 10, 1, 3, -4, 0, -4, 1, -4, 2, -4, 6, -1, 5, -1, 4, -1, 7, -1,
                                 5, -4, 6, -4, 7, -4, 4, -4, 10, -1, 9, -1, 8, -1, 11, -1, 1, 3, 0, 3, 3, 3, 2, 3,
                                 7, -2, 4, -2, 5, -2, 6, -2, 11, 2, 8, 2, 9, 2, 10, 2, 2, -4, 3, -4, 0, -4, 1, -4,
                                 10, -4, 11, -4, 8, -4, 9, -4, 1, -2, 2, -2, 3, -2, 0, -2, 5, 2, 6, 2, 7, 2, 4, 2,
                                 11, 3, 10, 3, 9, 3, 8, 3, 2, 1, 1, 1, 0, 1, 3, 1, 7, 0, 4, 0, 5, 0, 6, 0,
                                 6, 3, 5, 3, 4, 3, 7, 3, 9, 0, 10, 0, 11, 0, 8, 0, 3, 1, 2, 1, 1, 1, 0, 1};
  static PetscInt hex_hex[]   = {3, -24, 0, -24, 4, -24, 5, -24, 2, -24, 6, -24, 7, -24, 1, -24,
                                 3, -23, 5, -23, 6, -23, 2, -23, 0, -23, 1, -23, 7, -23, 4, -23,
                                 4, -22, 0, -22, 1, -22, 7, -22, 5, -22, 6, -22, 2, -22, 3, -22,
                                 6, -21, 7, -21, 1, -21, 2, -21, 5, -21, 3, -21, 0, -21, 4, -21,
                                 1, -20, 2, -20, 6, -20, 7, -20, 0, -20, 4, -20, 5, -20, 3, -20,
                                 6, -19, 2, -19, 3, -19, 5, -19, 7, -19, 4, -19, 0, -19, 1, -19,
                                 4, -18, 5, -18, 3, -18, 0, -18, 7, -18, 1, -18, 2, -18, 6, -18,
                                 1, -17, 7, -17, 4, -17, 0, -17, 2, -17, 3, -17, 5, -17, 6, -17,
                                 2, -16, 3, -16, 5, -16, 6, -16, 1, -16, 7, -16, 4, -16, 0, -16,
                                 7, -15, 4, -15, 0, -15, 1, -15, 6, -15, 2, -15, 3, -15, 5, -15,
                                 7, -14, 1, -14, 2, -14, 6, -14, 4, -14, 5, -14, 3, -14, 0, -14,
                                 0, -13, 4, -13, 5, -13, 3, -13, 1, -13, 2, -13, 6, -13, 7, -13,
                                 5, -12, 4, -12, 7, -12, 6, -12, 3, -12, 2, -12, 1, -12, 0, -12,
                                 7, -11, 6, -11, 5, -11, 4, -11, 1, -11, 0, -11, 3, -11, 2, -11,
                                 0, -10, 1, -10, 7, -10, 4, -10, 3, -10, 5, -10, 6, -10, 2, -10,
                                 4, -9, 7, -9, 6, -9, 5, -9, 0, -9, 3, -9, 2, -9, 1, -9,
                                 5, -8, 6, -8, 2, -8, 3, -8, 4, -8, 0, -8, 1, -8, 7, -8,
                                 2, -7, 6, -7, 7, -7, 1, -7, 3, -7, 0, -7, 4, -7, 5, -7,
                                 6, -6, 5, -6, 4, -6, 7, -6, 2, -6, 1, -6, 0, -6, 3, -6,
                                 5, -5, 3, -5, 0, -5, 4, -5, 6, -5, 7, -5, 1, -5, 2, -5,
                                 2, -4, 1, -4, 0, -4, 3, -4, 6, -4, 5, -4, 4, -4, 7, -4,
                                 1, -3, 0, -3, 3, -3, 2, -3, 7, -3, 6, -3, 5, -3, 4, -3,
                                 0, -2, 3, -2, 2, -2, 1, -2, 4, -2, 7, -2, 6, -2, 5, -2,
                                 3, -1, 2, -1, 1, -1, 0, -1, 5, -1, 4, -1, 7, -1, 6, -1,
                                 0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
                                 1, 1, 2, 1, 3, 1, 0, 1, 7, 1, 4, 1, 5, 1, 6, 1,
                                 2, 2, 3, 2, 0, 2, 1, 2, 6, 2, 7, 2, 4, 2, 5, 2,
                                 3, 3, 0, 3, 1, 3, 2, 3, 5, 3, 6, 3, 7, 3, 4, 3,
                                 4, 4, 0, 4, 3, 4, 5, 4, 7, 4, 6, 4, 2, 4, 1, 4,
                                 7, 5, 4, 5, 5, 5, 6, 5, 1, 5, 2, 5, 3, 5, 0, 5,
                                 1, 6, 7, 6, 6, 6, 2, 6, 0, 6, 3, 6, 5, 6, 4, 6,
                                 3, 7, 2, 7, 6, 7, 5, 7, 0, 7, 4, 7, 7, 7, 1, 7,
                                 5, 8, 6, 8, 7, 8, 4, 8, 3, 8, 0, 8, 1, 8, 2, 8,
                                 4, 9, 7, 9, 1, 9, 0, 9, 5, 9, 3, 9, 2, 9, 6, 9,
                                 4, 10, 5, 10, 6, 10, 7, 10, 0, 10, 1, 10, 2, 10, 3, 10,
                                 6, 11, 7, 11, 4, 11, 5, 11, 2, 11, 3, 11, 0, 11, 1, 11,
                                 3, 12, 5, 12, 4, 12, 0, 12, 2, 12, 1, 12, 7, 12, 6, 12,
                                 6, 13, 2, 13, 1, 13, 7, 13, 5, 13, 4, 13, 0, 13, 3, 13,
                                 1, 14, 0, 14, 4, 14, 7, 14, 2, 14, 6, 14, 5, 14, 3, 14,
                                 6, 15, 5, 15, 3, 15, 2, 15, 7, 15, 1, 15, 0, 15, 4, 15,
                                 0, 16, 4, 16, 7, 16, 1, 16, 3, 16, 2, 16, 6, 16, 5, 16,
                                 0, 17, 3, 17, 5, 17, 4, 17, 1, 17, 7, 17, 6, 17, 2, 17,
                                 5, 18, 3, 18, 2, 18, 6, 18, 4, 18, 7, 18, 1, 18, 0, 18,
                                 7, 19, 6, 19, 2, 19, 1, 19, 4, 19, 0, 19, 3, 19, 5, 19,
                                 2, 20, 1, 20, 7, 20, 6, 20, 3, 20, 5, 20, 4, 20, 0, 20,
                                 7, 21, 1, 21, 0, 21, 4, 21, 6, 21, 5, 21, 3, 21, 2, 21,
                                 2, 22, 6, 22, 5, 22, 3, 22, 1, 22, 0, 22, 4, 22, 7, 22,
                                 5, 23, 4, 23, 0, 23, 3, 23, 6, 23, 2, 23, 1, 23, 7, 23};
  static PetscInt trip_seg[]   = {1,  0, 2,  0, 0,  0,
                                  2,  0, 0,  0, 1,  0,
                                  0,  0, 1,  0, 2,  0,
                                  0, -1, 2, -1, 1, -1,
                                  1, -1, 0, -1, 2, -1,
                                  2, -1, 1, -1, 0, -1,
                                  0,  0, 1,  0, 2,  0,
                                  2,  0, 0,  0, 1,  0,
                                  1,  0, 2,  0, 0,  0,
                                  2, -1, 1, -1, 0, -1,
                                  1, -1, 0, -1, 2, -1,
                                  0, -1, 2, -1, 1, -1};
  static PetscInt trip_tri[]   = {1, 1, 2, 1, 0, 1, 3, 1,
                                  2, 2, 0, 2, 1, 2, 3, 2,
                                  0, 0, 1, 0, 2, 0, 3, 0,
                                  2, -1, 1, -1, 0, -1, 3, -3,
                                  0, -2, 2, -2, 1, -2, 3, -1,
                                  1, -3, 0, -3, 2, -3, 3, -2,
                                  0, 0, 1, 0, 2, 0, 3, 0,
                                  2, 2, 0, 2, 1, 2, 3, 2,
                                  1, 1, 2, 1, 0, 1, 3, 1,
                                  1, -3, 0, -3, 2, -3, 3, -2,
                                  0, -2, 2, -2, 1, -2, 3, -1,
                                  2, -1, 1, -1, 0, -1, 3, -3};
  static PetscInt trip_quad[]  = {4, -1, 5, -1, 3, -1, 1, -1, 2, -1, 0, -1,
                                  5, -1, 3, -1, 4, -1, 2, -1, 0, -1, 1, -1,
                                  3, -1, 4, -1, 5, -1, 0, -1, 1, -1, 2, -1,
                                  0, -3, 2, -3, 1, -3, 3, -3, 5, -3, 4, -3,
                                  1, -3, 0, -3, 2, -3, 4, -3, 3, -3, 5, -3,
                                  2, -3, 1, -3, 0, -3, 5, -3, 4, -3, 3, -3,
                                  0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0,
                                  2, 0, 0, 0, 1, 0, 5, 0, 3, 0, 4, 0,
                                  1, 0, 2, 0, 0, 0, 4, 0, 5, 0, 3, 0,
                                  5, 2, 4, 2, 3, 2, 2, 2, 1, 2, 0, 2,
                                  4, 2, 3, 2, 5, 2, 1, 2, 0, 2, 2, 2,
                                  3, 2, 5, 2, 4, 2, 0, 2, 2, 2, 1, 2};
  static PetscInt trip_trip[]  = {5, -6, 6, -6, 4, -6, 7, -6, 1, -6, 2, -6, 0, -6, 3, -6,
                                  6, -5, 4, -5, 5, -5, 7, -5, 2, -5, 0, -5, 1, -5, 3, -5,
                                  4, -4, 5, -4, 6, -4, 7, -4, 0, -4, 1, -4, 2, -4, 3, -4,
                                  2, -3, 1, -3, 0, -3, 3, -1, 6, -3, 5, -3, 4, -3, 7, -1,
                                  0, -2, 2, -2, 1, -2, 3, -3, 4, -2, 6, -2, 5, -2, 7, -3,
                                  1, -1, 0, -1, 2, -1, 3, -2, 5, -1, 4, -1, 6, -1, 7, -2,
                                  0, 0, 1, 0, 2, 0, 3, 0, 4, 0, 5, 0, 6, 0, 7, 0,
                                  2, 1, 0, 1, 1, 1, 3, 1, 6, 1, 4, 1, 5, 1, 7, 1,
                                  1, 2, 2, 2, 0, 2, 3, 2, 5, 2, 6, 2, 4, 2, 7, 2,
                                  5, 3, 4, 3, 6, 3, 7, 4, 1, 3, 0, 3, 2, 3, 3, 4,
                                  4, 4, 6, 4, 5, 4, 7, 5, 0, 4, 2, 4, 1, 4, 3, 5,
                                  6, 5, 5, 5, 4, 5, 7, 3, 2, 5, 1, 5, 0, 5, 3, 3};
  static PetscInt ttri_tseg[]  = {2, -2, 1, -2, 0, -2,
                                  1, -2, 0, -2, 2, -2,
                                  0, -2, 2, -2, 1, -2,
                                  2, -1, 1, -1, 0, -1,
                                  1, -1, 0, -1, 2, -1,
                                  0, -1, 2, -1, 1, -1,
                                  0, 0, 1, 0, 2, 0,
                                  1, 0, 2, 0, 0, 0,
                                  2, 0, 0, 0, 1, 0,
                                  0, 1, 1, 1, 2, 1,
                                  1, 1, 2, 1, 0, 1,
                                  2, 1, 0, 1, 1, 1};
  static PetscInt ttri_ttri[]  = {1, -6, 0, -6, 2, -6, 3, -5,
                                  0, -5, 2, -5, 1, -5, 3, -4,
                                  2, -4, 1, -4, 0, -4, 3, -6,
                                  1, -3, 0, -3, 2, -3, 3, -2,
                                  0, -2, 2, -2, 1, -2, 3, -1,
                                  2, -1, 1, -1, 0, -1, 3, -3,
                                  0, 0, 1, 0, 2, 0, 3, 0,
                                  1, 1, 2, 1, 0, 1, 3, 1,
                                  2, 2, 0, 2, 1, 2, 3, 2,
                                  0, 3, 1, 3, 2, 3, 3, 3,
                                  1, 4, 2, 4, 0, 4, 3, 4,
                                  2, 5, 0, 5, 1, 5, 3, 5};
  static PetscInt tquad_tvert[]  = {0, -1,
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
  static PetscInt tquad_tseg[]   = {1, 1, 0, 1, 3, 1, 2, 1,
                                    0, 1, 3, 1, 2, 1, 1, 1,
                                    3, 1, 2, 1, 1, 1, 0, 1,
                                    2, 1, 1, 1, 0, 1, 3, 1,
                                    1, 0, 0, 0, 3, 0, 2, 0,
                                    0, 0, 3, 0, 2, 0, 1, 0,
                                    3, 0, 2, 0, 1, 0, 0, 0,
                                    2, 0, 1, 0, 0, 0, 3, 0,
                                    0, 0, 1, 0, 2, 0, 3, 0,
                                    1, 0, 2, 0, 3, 0, 0, 0,
                                    2, 0, 3, 0, 0, 0, 1, 0,
                                    3, 0, 0, 0, 1, 0, 2, 0,
                                    0, 1, 1, 1, 2, 1, 3, 1,
                                    1, 1, 2, 1, 3, 1, 0, 1,
                                    2, 1, 3, 1, 0, 1, 1, 1,
                                    3, 1, 0, 1, 1, 1, 2, 1};
  static PetscInt tquad_tquad[]  = {2, -8, 1, -8, 0, -8, 3, -8,
                                    1, -7, 0, -7, 3, -7, 2, -7,
                                    0, -6, 3, -6, 2, -6, 1, -6,
                                    3, -5, 2, -5, 1, -5, 0, -5,
                                    2, -4, 1, -4, 0, -4, 3, -4,
                                    1, -3, 0, -3, 3, -3, 2, -3,
                                    0, -2, 3, -2, 2, -2, 1, -2,
                                    3, -1, 2, -1, 1, -1, 0, -1,
                                    0, 0, 1, 0, 2, 0, 3, 0,
                                    1, 1, 2, 1, 3, 1, 0, 1,
                                    2, 2, 3, 2, 0, 2, 1, 2,
                                    3, 3, 0, 3, 1, 3, 2, 3,
                                    0, 4, 1, 4, 2, 4, 3, 4,
                                    1, 5, 2, 5, 3, 5, 0, 5,
                                    2, 6, 3, 6, 0, 6, 1, 6,
                                    3, 7, 0, 7, 1, 7, 2, 7};

  PetscFunctionBeginHot;
  *rnew = r; *onew = o;
  if (!so) PetscFunctionReturn(0);
  switch (sct) {
    case DM_POLYTOPE_POINT: break;
    case DM_POLYTOPE_POINT_PRISM_TENSOR:
    *onew = so < 0 ? -(o+1) : o;
    break;
    case DM_POLYTOPE_SEGMENT:
    switch (tct) {
      case DM_POLYTOPE_POINT: break;
      case DM_POLYTOPE_SEGMENT:
        *rnew = seg_seg[(so+1)*4 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, seg_seg[(so+1)*4 + r*2 + 1]);
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
    case DM_POLYTOPE_TRIANGLE:
    switch (tct) {
      case DM_POLYTOPE_SEGMENT:
        *rnew = tri_seg[(so+3)*6 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, tri_seg[(so+3)*6 + r*2 + 1]);
        break;
      case DM_POLYTOPE_TRIANGLE:
        *rnew = tri_tri[(so+3)*8 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, tri_tri[(so+3)*8 + r*2 + 1]);
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
    case DM_POLYTOPE_QUADRILATERAL:
    switch (tct) {
      case DM_POLYTOPE_POINT: break;
      case DM_POLYTOPE_SEGMENT:
        *rnew = quad_seg[(so+4)*8 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, quad_seg[(so+4)*8 + r*2 + 1]);
        break;
      case DM_POLYTOPE_QUADRILATERAL:
        *rnew = quad_quad[(so+4)*8 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, quad_quad[(so+4)*8 + r*2 + 1]);
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:
    switch (tct) {
      case DM_POLYTOPE_POINT_PRISM_TENSOR:
        *rnew = tseg_seg[(so+2)*2 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, tseg_seg[(so+2)*2 + r*2 + 1]);
        break;
      case DM_POLYTOPE_SEG_PRISM_TENSOR:
        *rnew = tseg_tseg[(so+2)*4 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, tseg_tseg[(so+2)*4 + r*2 + 1]);
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
    case DM_POLYTOPE_TETRAHEDRON:
    switch (tct) {
      case DM_POLYTOPE_SEGMENT:
        *rnew = tet_seg[(so+12)*2 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, tet_seg[(so+12)*2 + r*2 + 1]);
        break;
      case DM_POLYTOPE_TRIANGLE:
        *rnew = tet_tri[(so+12)*16 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, tet_tri[(so+12)*16 + r*2 + 1]);
        break;
      case DM_POLYTOPE_TETRAHEDRON:
        *rnew = tet_tet[(so+12)*16 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, tet_tet[(so+12)*16 + r*2 + 1]);
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
    case DM_POLYTOPE_HEXAHEDRON:
    switch (tct) {
      case DM_POLYTOPE_POINT: break;
      case DM_POLYTOPE_SEGMENT:
        *rnew = hex_seg[(so+24)*12 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, hex_seg[(so+24)*12 + r*2 + 1]);
        break;
      case DM_POLYTOPE_QUADRILATERAL:
        *rnew = hex_quad[(so+24)*24 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, hex_quad[(so+24)*24 + r*2 + 1]);
        break;
      case DM_POLYTOPE_HEXAHEDRON:
        *rnew = hex_hex[(so+24)*16 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, hex_hex[(so+24)*16 + r*2 + 1]);
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
    case DM_POLYTOPE_TRI_PRISM:
    switch (tct) {
      case DM_POLYTOPE_SEGMENT:
        *rnew = trip_seg[(so+6)*6 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, trip_seg[(so+6)*6 + r*2 + 1]);
        break;
      case DM_POLYTOPE_TRIANGLE:
        *rnew = trip_tri[(so+6)*8 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, trip_tri[(so+6)*8 + r*2 + 1]);
        break;
      case DM_POLYTOPE_QUADRILATERAL:
        *rnew = trip_quad[(so+6)*12 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, trip_quad[(so+6)*12 + r*2 + 1]);
        break;
      case DM_POLYTOPE_TRI_PRISM:
        *rnew = trip_trip[(so+6)*16 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, trip_trip[(so+6)*16 + r*2 + 1]);
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
    case DM_POLYTOPE_TRI_PRISM_TENSOR:
    switch (tct) {
      case DM_POLYTOPE_SEG_PRISM_TENSOR:
        *rnew = ttri_tseg[(so+6)*6 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, ttri_tseg[(so+6)*6 + r*2 + 1]);
        break;
      case DM_POLYTOPE_TRI_PRISM_TENSOR:
        *rnew = ttri_ttri[(so+6)*8 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, ttri_ttri[(so+6)*8 + r*2 + 1]);
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:
    switch (tct) {
      case DM_POLYTOPE_POINT_PRISM_TENSOR:
        *rnew = tquad_tvert[(so+8)*2 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, tquad_tvert[(so+8)*2 + r*2 + 1]);
        break;
      case DM_POLYTOPE_SEG_PRISM_TENSOR:
        *rnew = tquad_tseg[(so+8)*8 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, tquad_tseg[(so+8)*8 + r*2 + 1]);
        break;
      case DM_POLYTOPE_QUAD_PRISM_TENSOR:
        *rnew = tquad_tquad[(so+8)*8 + r*2];
        *onew = DMPolytopeTypeComposeOrientation(tct, o, tquad_tquad[(so+8)*8 + r*2 + 1]);
        break;
      default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "Cell type %s is not produced by %s", DMPolytopeTypes[tct], DMPolytopeTypes[sct]);
    }
    break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_SUP, "Unsupported cell type %s", DMPolytopeTypes[sct]);
  }
  PetscFunctionReturn(0);
}

PetscErrorCode DMPlexTransformCellRefine_Regular(DMPlexTransform tr, DMPolytopeType source, PetscInt p, PetscInt *rt, PetscInt *Nt, DMPolytopeType *target[], PetscInt *size[], PetscInt *cone[], PetscInt *ornt[])
{
  /* All vertices remain in the refined mesh */
  static DMPolytopeType vertexT[] = {DM_POLYTOPE_POINT};
  static PetscInt       vertexS[] = {1};
  static PetscInt       vertexC[] = {0};
  static PetscInt       vertexO[] = {0};
  /* Split all edges with a new vertex, making two new 2 edges
     0--0--0--1--1
  */
  static DMPolytopeType segT[]    = {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT};
  static PetscInt       segS[]    = {1, 2};
  static PetscInt       segC[]    = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 0, 0,  DM_POLYTOPE_POINT, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0};
  static PetscInt       segO[]    = {                         0,                       0,                        0,                          0};
  /* Do not split tensor edges */
  static DMPolytopeType tvertT[]  = {DM_POLYTOPE_POINT_PRISM_TENSOR};
  static PetscInt       tvertS[]  = {1};
  static PetscInt       tvertC[]  = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0};
  static PetscInt       tvertO[]  = {                         0,                          0};
  /* Add 3 edges inside every triangle, making 4 new triangles.
   2
   |\
   | \
   |  \
   0   1
   | C  \
   |     \
   |      \
   2---1---1
   |\  D  / \
   1 2   0   0
   |A \ /  B  \
   0-0-0---1---1
  */
  static DMPolytopeType triT[]    = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE};
  static PetscInt       triS[]    = {3, 4};
  static PetscInt       triC[]    = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0,
                                     DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 1, 2, 0,
                                     DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_POINT, 1, 0, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 1, 2, 1,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0,    0,
                                     DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0,
                                     DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 0,    2};
  static PetscInt       triO[]    = {0, 0,
                                     0, 0,
                                     0, 0,
                                     0, -1,  0,
                                     0,  0, -1,
                                    -1,  0,  0,
                                     0,  0,  0};
  /* Add a vertex in the center of each quadrilateral, and 4 edges inside, making 4 new quads.
     3----1----2----0----2
     |         |         |
     0    D    2    C    1
     |         |         |
     3----3----0----1----1
     |         |         |
     1    A    0    B    0
     |         |         |
     0----0----0----1----1
  */
  static DMPolytopeType quadT[]   = {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT, DM_POLYTOPE_QUADRILATERAL};
  static PetscInt       quadS[]   = {1, 4, 4};
  static PetscInt       quadC[]   = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 3, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 1, 3, 1,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 0,    0,
                                     DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 0,    2,
                                     DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 1, 3, 0};
  static PetscInt       quadO[]   = {0, 0,
                                     0, 0,
                                     0, 0,
                                     0, 0,
                                     0,  0, -1,  0,
                                     0,  0,  0, -1,
                                    -1,  0,  0,  0,
                                     0, -1,  0,  0};
  /* Add 1 edge inside every tensor quad, making 2 new tensor quads
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
  static DMPolytopeType tsegT[]  = {DM_POLYTOPE_POINT_PRISM_TENSOR, DM_POLYTOPE_SEG_PRISM_TENSOR};
  static PetscInt       tsegS[]  = {1, 2};
  static PetscInt       tsegC[]  = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0,
                                    DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 2, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 0,   0,
                                    DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_POINT_PRISM_TENSOR, 0,    0, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 3, 0};
  static PetscInt       tsegO[]  = {0, 0,
                                    0, 0, 0, 0,
                                    0, 0, 0, 0};
  /* Add 1 edge and 8 triangles inside every cell, making 8 new tets
     The vertices of our reference tet are [(-1, -1, -1), (-1, 1, -1), (1, -1, -1), (-1, -1, 1)], which we call [v0, v1, v2, v3]. The first
     three edges are [v0, v1], [v1, v2], [v2, v0] called e0, e1, and e2, and then three edges to the top point [v0, v3], [v1, v3], [v2, v3]
     called e3, e4, and e5. The faces of a tet, given in DMPlexGetRawFaces_Internal() are
       [v0, v1, v2], [v0, v3, v1], [v0, v2, v3], [v2, v1, v3]
     The first four tets just cut off the corners, using the replica notation for new vertices,
       [v0,      (e0, 0), (e2, 0), (e3, 0)]
       [(e0, 0), v1,      (e1, 0), (e4, 0)]
       [(e2, 0), (e1, 0), v2,      (e5, 0)]
       [(e3, 0), (e4, 0), (e5, 0), v3     ]
     The next four tets match a vertex to the newly created faces from cutting off those first tets.
       [(e2, 0), (e3, 0), (e0, 0), (e5, 0)]
       [(e4, 0), (e1, 0), (e0, 0), (e5, 0)]
       [(e5, 0), (e0, 0), (e2, 0), (e1, 0)]
       [(e5, 0), (e0, 0), (e4, 0), (e3, 0)]
     We can see that a new edge is introduced in the cell [(e0, 0), (e5, 0)] which we call (-1, 0). The first four faces created are
       [(e2, 0), (e0, 0), (e3, 0)]
       [(e0, 0), (e1, 0), (e4, 0)]
       [(e2, 0), (e5, 0), (e1, 0)]
       [(e3, 0), (e4, 0), (e5, 0)]
     The next four, from the second group of tets, are
       [(e2, 0), (e0, 0), (e5, 0)]
       [(e4, 0), (e0, 0), (e5, 0)]
       [(e0, 0), (e1, 0), (e5, 0)]
       [(e5, 0), (e3, 0), (e0, 0)]
     I could write a program to generate these orientations by comparing the faces from GetRawFaces() with my existing table.
   */
  static DMPolytopeType tetT[]    = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_TETRAHEDRON};
  static PetscInt       tetS[]    = {1, 8, 8};
  static PetscInt       tetC[]    = {DM_POLYTOPE_POINT, 2, 0, 0, 0, DM_POLYTOPE_POINT, 2, 2, 1, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_SEGMENT, 1, 2, 2,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 3, 0, DM_POLYTOPE_SEGMENT, 1, 1, 1,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 1, 3, 2, DM_POLYTOPE_SEGMENT, 1, 0, 1,
                                     DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 3, 1, DM_POLYTOPE_SEGMENT, 1, 2, 1,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 2, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 3, 1,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 3, 2, DM_POLYTOPE_SEGMENT, 0,    0,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_SEGMENT, 0,    0,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 0, DM_POLYTOPE_TRIANGLE, 1, 1, 0, DM_POLYTOPE_TRIANGLE, 1, 2, 0, DM_POLYTOPE_TRIANGLE, 0,    0,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 1, DM_POLYTOPE_TRIANGLE, 1, 1, 2, DM_POLYTOPE_TRIANGLE, 0,    1, DM_POLYTOPE_TRIANGLE, 1, 3, 1,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 2, DM_POLYTOPE_TRIANGLE, 0,    2, DM_POLYTOPE_TRIANGLE, 1, 2, 1, DM_POLYTOPE_TRIANGLE, 1, 3, 0,
                                     DM_POLYTOPE_TRIANGLE, 0,    3, DM_POLYTOPE_TRIANGLE, 1, 1, 1, DM_POLYTOPE_TRIANGLE, 1, 2, 2, DM_POLYTOPE_TRIANGLE, 1, 3, 2,
                                     DM_POLYTOPE_TRIANGLE, 0,    0, DM_POLYTOPE_TRIANGLE, 1, 2, 3, DM_POLYTOPE_TRIANGLE, 0,    4, DM_POLYTOPE_TRIANGLE, 0,    7,
                                     DM_POLYTOPE_TRIANGLE, 0,    1, DM_POLYTOPE_TRIANGLE, 1, 3, 3, DM_POLYTOPE_TRIANGLE, 0,    5, DM_POLYTOPE_TRIANGLE, 0,    6,
                                     DM_POLYTOPE_TRIANGLE, 0,    4, DM_POLYTOPE_TRIANGLE, 0,    6, DM_POLYTOPE_TRIANGLE, 0,    2, DM_POLYTOPE_TRIANGLE, 1, 0, 3,
                                     DM_POLYTOPE_TRIANGLE, 0,    5, DM_POLYTOPE_TRIANGLE, 0,    7, DM_POLYTOPE_TRIANGLE, 0,    3, DM_POLYTOPE_TRIANGLE, 1, 1, 3};
  static PetscInt       tetO[]    = {0, 0,
                                     0,  0,  0,
                                     0,  0,  0,
                                     0,  0,  0,
                                     0,  0,  0,
                                     0,  0, -1,
                                     0,  0, -1,
                                     0, -1, -1,
                                     0, -1,  0,
                                     0,  0,  0,  0,
                                     0,  0,  0,  0,
                                     0,  0,  0,  0,
                                     0,  0,  0,  0,
                                    -2,  0,  0, -1,
                                    -1,  1,  0,  0,
                                    -1, -1, -3,  2,
                                    -1,  0, -1,  1};
  /* Add a vertex in the center of each cell, add 6 edges and 12 quads inside every cell, making 8 new hexes
     The vertices of our reference hex are (-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1), (-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1) which we call [v0, v1, v2, v3, v4, v5, v6, v7]. The fours edges around the bottom [v0, v1], [v1, v2], [v2, v3], [v3, v0] are [e0, e1, e2, e3], and likewise around the top [v4, v5], [v5, v6], [v6, v7], [v7, v4] are [e4, e5, e6, e7]. Finally [v0, v4], [v1, v7], [v2, v6], [v3, v5] are [e9, e10, e11, e8]. The faces of a hex, given in DMPlexGetRawFaces_Internal(), oriented with outward normals, are
       [v0, v1, v2, v3] f0 bottom
       [v4, v5, v6, v7] f1 top
       [v0, v3, v5, v4] f2 front
       [v2, v1, v7, v6] f3 back
       [v3, v2, v6, v5] f4 right
       [v0, v4, v7, v1] f5 left
     The eight hexes are divided into four on the bottom, and four on the top,
       [v0,      (e0, 0),  (f0, 0),  (e3, 0),  (e9, 0), (f2, 0),  (c0, 0),  (f5, 0)]
       [(e0, 0), v1,       (e1, 0),  (f0, 0),  (f5, 0), (c0, 0),  (f3, 0),  (e10, 0)]
       [(f0, 0), (e1, 0),  v2,       (e2, 0),  (c0, 0), (f4, 0),  (e11, 0), (f3, 0)]
       [(e3, 0), (f0, 0),  (e2, 0),  v3,       (f2, 0), (e8, 0),  (f4, 0),  (c0, 0)]
       [(e9, 0), (f5, 0),  (c0, 0),  (f2, 0),  v4,      (e4, 0),  (f1, 0),  (e7, 0)]
       [(f2, 0), (c0, 0),  (f4, 0),  (e8, 0),  (e4, 0), v5,       (e5, 0),  (f1, 0)]
       [(c0, 0), (f3, 0),  (e11, 0), (f4, 0),  (f1, 0), (e5, 0),  v6,       (e6, 0)]
       [(f5, 0), (e10, 0), (f3, 0),  (c0, 0),  (e7, 0), (f1, 0),  (e6, 0),  v7]
     The 6 internal edges will go from the faces to the central vertex. The 12 internal faces can be divided into groups of 4 by the plane on which they sit. First the faces on the x-y plane are,
       [(e9, 0), (f2, 0),  (c0, 0),  (f5, 0)]
       [(f5, 0), (c0, 0),  (f3, 0),  (e10, 0)]
       [(c0, 0), (f4, 0),  (e11, 0), (f3, 0)]
       [(f2, 0), (e8, 0),  (f4, 0),  (c0, 0)]
     and on the x-z plane,
       [(f0, 0), (e0, 0), (f5, 0), (c0, 0)]
       [(c0, 0), (f5, 0), (e7, 0), (f1, 0)]
       [(f4, 0), (c0, 0), (f1, 0), (e5, 0)]
       [(e2, 0), (f0, 0), (c0, 0), (f4, 0)]
     and on the y-z plane,
       [(e3, 0), (f2, 0), (c0, 0), (f0, 0)]
       [(f2, 0), (e4, 0), (f1, 0), (c0, 0)]
       [(c0, 0), (f1, 0), (e6, 0), (f3, 0)]
       [(f0, 0), (c0, 0), (f3, 0), (e1, 0)]
  */
  static DMPolytopeType hexT[]    = {DM_POLYTOPE_POINT, DM_POLYTOPE_SEGMENT, DM_POLYTOPE_QUADRILATERAL, DM_POLYTOPE_HEXAHEDRON};
  static PetscInt       hexS[]    = {1, 6, 12, 8};
  static PetscInt       hexC[]    = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 1, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 3, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 4, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_POINT, 1, 5, 0, DM_POLYTOPE_POINT, 0, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 3, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 0,    5, DM_POLYTOPE_SEGMENT, 1, 5, 0,
                                     DM_POLYTOPE_SEGMENT, 0,    5, DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 1, 3, 1, DM_POLYTOPE_SEGMENT, 1, 5, 2,
                                     DM_POLYTOPE_SEGMENT, 0,    4, DM_POLYTOPE_SEGMENT, 1, 4, 1, DM_POLYTOPE_SEGMENT, 1, 3, 3, DM_POLYTOPE_SEGMENT, 0,    3,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 1, 4, 3, DM_POLYTOPE_SEGMENT, 0,    4, DM_POLYTOPE_SEGMENT, 0,    2,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 5, 3, DM_POLYTOPE_SEGMENT, 0,    5, DM_POLYTOPE_SEGMENT, 0,    0,
                                     DM_POLYTOPE_SEGMENT, 0,    5, DM_POLYTOPE_SEGMENT, 1, 5, 1, DM_POLYTOPE_SEGMENT, 1, 1, 3, DM_POLYTOPE_SEGMENT, 0,    1,
                                     DM_POLYTOPE_SEGMENT, 0,    4, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 4, 2,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    4, DM_POLYTOPE_SEGMENT, 1, 4, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 0, 3,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 2, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 0,    2,
                                     DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_SEGMENT, 1, 3, 2, DM_POLYTOPE_SEGMENT, 0,    3,
                                     DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 1, 3, 0, DM_POLYTOPE_SEGMENT, 1, 0, 1,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_QUADRILATERAL, 1, 2, 0, DM_POLYTOPE_QUADRILATERAL, 0,    4, DM_POLYTOPE_QUADRILATERAL, 0,    8, DM_POLYTOPE_QUADRILATERAL, 1, 5, 0,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 1, DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 0,    4, DM_POLYTOPE_QUADRILATERAL, 1, 3, 1, DM_POLYTOPE_QUADRILATERAL, 0,   11, DM_POLYTOPE_QUADRILATERAL, 1, 5, 3,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 2, DM_POLYTOPE_QUADRILATERAL, 0,    2, DM_POLYTOPE_QUADRILATERAL, 0,    7, DM_POLYTOPE_QUADRILATERAL, 1, 3, 0, DM_POLYTOPE_QUADRILATERAL, 1, 4, 1, DM_POLYTOPE_QUADRILATERAL, 0,   11,
                                     DM_POLYTOPE_QUADRILATERAL, 1, 0, 3, DM_POLYTOPE_QUADRILATERAL, 0,    3, DM_POLYTOPE_QUADRILATERAL, 1, 2, 1, DM_POLYTOPE_QUADRILATERAL, 0,    7, DM_POLYTOPE_QUADRILATERAL, 1, 4, 0, DM_POLYTOPE_QUADRILATERAL, 0,    8,
                                     DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_QUADRILATERAL, 1, 1, 0, DM_POLYTOPE_QUADRILATERAL, 1, 2, 3, DM_POLYTOPE_QUADRILATERAL, 0,    5, DM_POLYTOPE_QUADRILATERAL, 0,    9, DM_POLYTOPE_QUADRILATERAL, 1, 5, 1,
                                     DM_POLYTOPE_QUADRILATERAL, 0,    3, DM_POLYTOPE_QUADRILATERAL, 1, 1, 1, DM_POLYTOPE_QUADRILATERAL, 1, 2, 2, DM_POLYTOPE_QUADRILATERAL, 0,    6, DM_POLYTOPE_QUADRILATERAL, 1, 4, 3, DM_POLYTOPE_QUADRILATERAL, 0,    9,
                                     DM_POLYTOPE_QUADRILATERAL, 0,    2, DM_POLYTOPE_QUADRILATERAL, 1, 1, 2, DM_POLYTOPE_QUADRILATERAL, 0,    6, DM_POLYTOPE_QUADRILATERAL, 1, 3, 3, DM_POLYTOPE_QUADRILATERAL, 1, 4, 2, DM_POLYTOPE_QUADRILATERAL, 0,   10,
                                     DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 1, 1, 3, DM_POLYTOPE_QUADRILATERAL, 0,    5, DM_POLYTOPE_QUADRILATERAL, 1, 3, 2, DM_POLYTOPE_QUADRILATERAL, 0,   10, DM_POLYTOPE_QUADRILATERAL, 1, 5, 2};
  static PetscInt       hexO[]    = {0, 0,
                                     0, 0,
                                     0, 0,
                                     0, 0,
                                     0, 0,
                                     0, 0,
                                     0,  0, -1, -1,
                                     0, -1, -1,  0,
                                    -1, -1,  0,  0,
                                    -1,  0,  0, -1,
                                    -1,  0,  0, -1,
                                    -1, -1,  0,  0,
                                     0, -1, -1,  0,
                                     0,  0, -1, -1,
                                     0,  0, -1, -1,
                                    -1,  0,  0, -1,
                                    -1, -1,  0,  0,
                                     0, -1, -1,  0,
                                     0, 0,  0, 0, -2, 0,
                                     0, 0, -3, 0, -2, 0,
                                     0, 0, -3, 0,  0, 0,
                                     0, 0,  0, 0,  0, 0,
                                    -2, 0,  0, 0, -2, 0,
                                    -2, 0,  0, 0,  0, 0,
                                    -2, 0, -3, 0,  0, 0,
                                    -2, 0, -3, 0, -2, 0};
  /* Add 3 quads inside every triangular prism, making 4 new prisms. */
  static DMPolytopeType tripT[]   = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_QUADRILATERAL, DM_POLYTOPE_TRI_PRISM};
  static PetscInt       tripS[]   = {3, 4, 6, 8};
  static PetscInt       tripC[]   = {DM_POLYTOPE_POINT, 1, 2, 0, DM_POLYTOPE_POINT, 1, 3, 0,
                                     DM_POLYTOPE_POINT, 1, 3, 0, DM_POLYTOPE_POINT, 1, 4, 0,
                                     DM_POLYTOPE_POINT, 1, 4, 0, DM_POLYTOPE_POINT, 1, 2, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 3, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 1, 4, 1,
                                     DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 1, 3, 3, DM_POLYTOPE_SEGMENT, 0,    0,
                                     DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 3, 1, DM_POLYTOPE_SEGMENT, 1, 4, 3,
                                     DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 0,    2,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 3, 0, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 2, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 4, 0, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 3, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 1, 4, 0,
                                     DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 3, 2, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 1, 2, 2,
                                     DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 4, 2, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 3, 2,
                                     DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 1, 2, 2, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_SEGMENT, 1, 4, 2,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 0, DM_POLYTOPE_TRIANGLE, 0,    0, DM_POLYTOPE_QUADRILATERAL, 1, 2, 0, DM_POLYTOPE_QUADRILATERAL, 0,    2, DM_POLYTOPE_QUADRILATERAL, 1, 4, 1,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 2, DM_POLYTOPE_TRIANGLE, 0,    1, DM_POLYTOPE_QUADRILATERAL, 1, 2, 1, DM_POLYTOPE_QUADRILATERAL, 1, 3, 0, DM_POLYTOPE_QUADRILATERAL, 0,    0,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 1, DM_POLYTOPE_TRIANGLE, 0,    2, DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 1, 3, 1, DM_POLYTOPE_QUADRILATERAL, 1, 4, 0,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 3, DM_POLYTOPE_TRIANGLE, 0,    3, DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_QUADRILATERAL, 0,    1, DM_POLYTOPE_QUADRILATERAL, 0,    2,
                                     DM_POLYTOPE_TRIANGLE, 0,    0, DM_POLYTOPE_TRIANGLE, 1, 1, 0, DM_POLYTOPE_QUADRILATERAL, 1, 2, 3, DM_POLYTOPE_QUADRILATERAL, 0,    5, DM_POLYTOPE_QUADRILATERAL, 1, 4, 2,
                                     DM_POLYTOPE_TRIANGLE, 0,    1, DM_POLYTOPE_TRIANGLE, 1, 1, 1, DM_POLYTOPE_QUADRILATERAL, 1, 2, 2, DM_POLYTOPE_QUADRILATERAL, 1, 3, 3, DM_POLYTOPE_QUADRILATERAL, 0,    3,
                                     DM_POLYTOPE_TRIANGLE, 0,    2, DM_POLYTOPE_TRIANGLE, 1, 1, 2, DM_POLYTOPE_QUADRILATERAL, 0,    4, DM_POLYTOPE_QUADRILATERAL, 1, 3, 2, DM_POLYTOPE_QUADRILATERAL, 1, 4, 3,
                                     DM_POLYTOPE_TRIANGLE, 0,    3, DM_POLYTOPE_TRIANGLE, 1, 1, 3, DM_POLYTOPE_QUADRILATERAL, 0,    3, DM_POLYTOPE_QUADRILATERAL, 0,    4, DM_POLYTOPE_QUADRILATERAL, 0,    5};
  static PetscInt       tripO[]   = {0, 0,
                                     0, 0,
                                     0, 0,
                                     0, -1, -1,
                                    -1,  0, -1,
                                    -1, -1,  0,
                                     0,  0,  0,
                                    -1,  0, -1, -1,
                                    -1,  0, -1, -1,
                                    -1,  0, -1, -1,
                                     0, -1, -1,  0,
                                     0, -1, -1,  0,
                                     0, -1, -1,  0,
                                     0,  0,  0, -3,  0,
                                     0,  0,  0,  0, -3,
                                     0,  0, -3,  0,  0,
                                     2,  0,  0,  0,  0,
                                    -2,  0,  0, -3,  0,
                                    -2,  0,  0,  0, -3,
                                    -2,  0, -3,  0,  0,
                                    -2,  0,  0,  0,  0};
  /* Add 3 tensor quads inside every tensor triangular prism, making 4 new prisms.
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
  static DMPolytopeType ttriT[]   = {DM_POLYTOPE_SEG_PRISM_TENSOR, DM_POLYTOPE_TRI_PRISM_TENSOR};
  static PetscInt       ttriS[]   = {3, 4};
  static PetscInt       ttriC[]   = {DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 2, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 3, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 3, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 4, 0,
                                     DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 4, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 2, 0,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 0, DM_POLYTOPE_TRIANGLE, 1, 1, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 2, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    2, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 4, 1,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 1, DM_POLYTOPE_TRIANGLE, 1, 1, 1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 2, 1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 3, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    0,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 2, DM_POLYTOPE_TRIANGLE, 1, 1, 2, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 3, 1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 4, 0,
                                     DM_POLYTOPE_TRIANGLE, 1, 0, 3, DM_POLYTOPE_TRIANGLE, 1, 1, 3, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    0, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,     1, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    2};
  static PetscInt       ttriO[]   = {0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0, 0, 0,
                                     0, 0,  0, -1,  0,
                                     0, 0,  0,  0, -1,
                                     0, 0, -1,  0,  0,
                                     0, 0,  0,  0,  0};
  /* Add 1 edge and 4 tensor quads inside every tensor quad prism, making 4 new prisms. */
  static DMPolytopeType tquadT[]   = {DM_POLYTOPE_POINT_PRISM_TENSOR, DM_POLYTOPE_SEG_PRISM_TENSOR, DM_POLYTOPE_QUAD_PRISM_TENSOR};
  static PetscInt       tquadS[]   = {1, 4, 4};
  static PetscInt       tquadC[]   = {DM_POLYTOPE_POINT, 1, 0, 0, DM_POLYTOPE_POINT, 1, 1, 0,
                                      DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 2, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 0, 0,
                                      DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 3, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 0, 0,
                                      DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 1, 1, 2, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 4, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 0, 0,
                                      DM_POLYTOPE_SEGMENT, 1, 0, 3, DM_POLYTOPE_SEGMENT, 1, 1, 3, DM_POLYTOPE_POINT_PRISM_TENSOR, 1, 5, 0, DM_POLYTOPE_POINT_PRISM_TENSOR, 0, 0,
                                      DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_QUADRILATERAL, 1, 1, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 2, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    0, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    3, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 5, 1,
                                      DM_POLYTOPE_QUADRILATERAL, 1, 0, 1, DM_POLYTOPE_QUADRILATERAL, 1, 1, 1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 2, 1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 3, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    1, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    0,
                                      DM_POLYTOPE_QUADRILATERAL, 1, 0, 2, DM_POLYTOPE_QUADRILATERAL, 1, 1, 2, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 3, 1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 4, 0, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    2,
                                      DM_POLYTOPE_QUADRILATERAL, 1, 0, 3, DM_POLYTOPE_QUADRILATERAL, 1, 1, 3, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    3, DM_POLYTOPE_SEG_PRISM_TENSOR, 0,    2, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 4, 1, DM_POLYTOPE_SEG_PRISM_TENSOR, 1, 5, 0};
  static PetscInt       tquadO[]   = {0, 0,
                                      0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      0, 0, 0, 0,
                                      0, 0,  0,  0, -1,  0,
                                      0, 0,  0,  0,  0, -1,
                                      0, 0, -1,  0,  0,  0,
                                      0, 0,  0, -1,  0,  0};
  /* Add 4 edges, 12 triangles, 1 quad, 4 tetrahedra, and 6 pyramids inside every pyramid. */
  static DMPolytopeType tpyrT[] = {DM_POLYTOPE_SEGMENT, DM_POLYTOPE_TRIANGLE, DM_POLYTOPE_QUADRILATERAL, DM_POLYTOPE_TETRAHEDRON, DM_POLYTOPE_PYRAMID};
  static PetscInt       tpyrS[] = {4, 12, 1, 4, 6};
  static PetscInt       tpyrC[] = {DM_POLYTOPE_POINT, 2, 1, 1, 0, DM_POLYTOPE_POINT, 1, 0, 0,
                                   DM_POLYTOPE_POINT, 2, 2, 1, 0, DM_POLYTOPE_POINT, 1, 0, 0,
                                   DM_POLYTOPE_POINT, 2, 3, 1, 0, DM_POLYTOPE_POINT, 1, 0, 0,
                                   DM_POLYTOPE_POINT, 2, 4, 1, 0, DM_POLYTOPE_POINT, 1, 0, 0,
                                   /* These four triangle face out of the bottom pyramid */
                                   DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 0, 3, DM_POLYTOPE_SEGMENT, 0, 0,
                                   DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 0, 0, DM_POLYTOPE_SEGMENT, 0, 1,
                                   DM_POLYTOPE_SEGMENT, 1, 3, 1, DM_POLYTOPE_SEGMENT, 0, 1, DM_POLYTOPE_SEGMENT, 0, 2,
                                   DM_POLYTOPE_SEGMENT, 1, 4, 1, DM_POLYTOPE_SEGMENT, 0, 2, DM_POLYTOPE_SEGMENT, 0, 3,
                                   /* These eight triangles face out of the corner pyramids */
                                   DM_POLYTOPE_SEGMENT, 1, 0, 3, DM_POLYTOPE_SEGMENT, 0,    3, DM_POLYTOPE_SEGMENT, 1, 1, 2,
                                   DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 0,    0, DM_POLYTOPE_SEGMENT, 1, 2, 2,
                                   DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 0,    1, DM_POLYTOPE_SEGMENT, 1, 3, 2,
                                   DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 0,    2, DM_POLYTOPE_SEGMENT, 1, 4, 2,
                                   DM_POLYTOPE_SEGMENT, 1, 0, 3, DM_POLYTOPE_SEGMENT, 1, 1, 0, DM_POLYTOPE_SEGMENT, 0,    0,
                                   DM_POLYTOPE_SEGMENT, 1, 0, 2, DM_POLYTOPE_SEGMENT, 1, 2, 0, DM_POLYTOPE_SEGMENT, 0,    1,
                                   DM_POLYTOPE_SEGMENT, 1, 0, 1, DM_POLYTOPE_SEGMENT, 1, 3, 0, DM_POLYTOPE_SEGMENT, 0,    2,
                                   DM_POLYTOPE_SEGMENT, 1, 0, 0, DM_POLYTOPE_SEGMENT, 1, 4, 0, DM_POLYTOPE_SEGMENT, 0,    3,
                                   /* This quad faces out of the bottom pyramid */
                                   DM_POLYTOPE_SEGMENT, 1, 1, 1, DM_POLYTOPE_SEGMENT, 1, 2, 1, DM_POLYTOPE_SEGMENT, 1, 3, 1, DM_POLYTOPE_SEGMENT, 1, 4, 1,
                                   /* The bottom face of each tet is on the triangular face */
                                   DM_POLYTOPE_TRIANGLE, 1, 1, 3, DM_POLYTOPE_TRIANGLE, 0,  8, DM_POLYTOPE_TRIANGLE, 0, 4, DM_POLYTOPE_TRIANGLE, 0, 0,
                                   DM_POLYTOPE_TRIANGLE, 1, 2, 3, DM_POLYTOPE_TRIANGLE, 0,  9, DM_POLYTOPE_TRIANGLE, 0, 5, DM_POLYTOPE_TRIANGLE, 0, 1,
                                   DM_POLYTOPE_TRIANGLE, 1, 3, 3, DM_POLYTOPE_TRIANGLE, 0, 10, DM_POLYTOPE_TRIANGLE, 0, 6, DM_POLYTOPE_TRIANGLE, 0, 2,
                                   DM_POLYTOPE_TRIANGLE, 1, 4, 3, DM_POLYTOPE_TRIANGLE, 0, 11, DM_POLYTOPE_TRIANGLE, 0, 7, DM_POLYTOPE_TRIANGLE, 0, 3,
                                   /* The front face of all pyramids is toward the front */
                                   DM_POLYTOPE_QUADRILATERAL, 1, 0, 0, DM_POLYTOPE_TRIANGLE, 1, 1, 0, DM_POLYTOPE_TRIANGLE, 0,    4, DM_POLYTOPE_TRIANGLE, 0,   11, DM_POLYTOPE_TRIANGLE, 1, 4, 1,
                                   DM_POLYTOPE_QUADRILATERAL, 1, 0, 3, DM_POLYTOPE_TRIANGLE, 1, 1, 1, DM_POLYTOPE_TRIANGLE, 1, 2, 0, DM_POLYTOPE_TRIANGLE, 0,    5, DM_POLYTOPE_TRIANGLE, 0,    8,
                                   DM_POLYTOPE_QUADRILATERAL, 1, 0, 2, DM_POLYTOPE_TRIANGLE, 0,    9, DM_POLYTOPE_TRIANGLE, 1, 2, 1, DM_POLYTOPE_TRIANGLE, 1, 3, 0, DM_POLYTOPE_TRIANGLE, 0,    6,
                                   DM_POLYTOPE_QUADRILATERAL, 1, 0, 1, DM_POLYTOPE_TRIANGLE, 0,    7, DM_POLYTOPE_TRIANGLE, 0,   10, DM_POLYTOPE_TRIANGLE, 1, 3, 1, DM_POLYTOPE_TRIANGLE, 1, 4, 0,
                                   DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_TRIANGLE, 1, 1, 2, DM_POLYTOPE_TRIANGLE, 1, 2, 2, DM_POLYTOPE_TRIANGLE, 1, 3, 2, DM_POLYTOPE_TRIANGLE, 1, 4, 2,
                                   DM_POLYTOPE_QUADRILATERAL, 0,    0, DM_POLYTOPE_TRIANGLE, 0,    0, DM_POLYTOPE_TRIANGLE, 0,    3, DM_POLYTOPE_TRIANGLE, 0,    2, DM_POLYTOPE_TRIANGLE, 0,    1,
                                   };
  static PetscInt       tpyrO[] = {0, 0,
                                   0, 0,
                                   0, 0,
                                   0, 0,
                                   0,  0, -1,
                                   0,  0, -1,
                                   0,  0, -1,
                                   0,  0, -1,
                                   0, -1,  0,
                                   0, -1,  0,
                                   0, -1,  0,
                                   0, -1,  0,
                                  -1,  0,  0,
                                  -1,  0,  0,
                                  -1,  0,  0,
                                  -1,  0,  0,
                                  -1, -1, -1, -1,
                                   0, -3, -2, -3,
                                   0, -3, -2, -3,
                                   0, -3, -2, -3,
                                   0, -3, -2, -3,
                                   0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0,
                                  -2, 0, 0, 0, 0,
                                   1, 0, 0, 0, 0};

  PetscFunctionBegin;
  if (rt) *rt = 0;
  switch (source) {
    case DM_POLYTOPE_POINT:              *Nt = 1; *target = vertexT; *size = vertexS; *cone = vertexC; *ornt = vertexO; break;
    case DM_POLYTOPE_SEGMENT:            *Nt = 2; *target = segT;    *size = segS;    *cone = segC;    *ornt = segO;    break;
    case DM_POLYTOPE_POINT_PRISM_TENSOR: *Nt = 1; *target = tvertT;  *size = tvertS;  *cone = tvertC;  *ornt = tvertO;  break;
    case DM_POLYTOPE_TRIANGLE:           *Nt = 2; *target = triT;    *size = triS;    *cone = triC;    *ornt = triO;    break;
    case DM_POLYTOPE_QUADRILATERAL:      *Nt = 3; *target = quadT;   *size = quadS;   *cone = quadC;   *ornt = quadO;   break;
    case DM_POLYTOPE_SEG_PRISM_TENSOR:   *Nt = 2; *target = tsegT;   *size = tsegS;   *cone = tsegC;   *ornt = tsegO;   break;
    case DM_POLYTOPE_TETRAHEDRON:        *Nt = 3; *target = tetT;    *size = tetS;    *cone = tetC;    *ornt = tetO;    break;
    case DM_POLYTOPE_HEXAHEDRON:         *Nt = 4; *target = hexT;    *size = hexS;    *cone = hexC;    *ornt = hexO;    break;
    case DM_POLYTOPE_TRI_PRISM:          *Nt = 4; *target = tripT;   *size = tripS;   *cone = tripC;   *ornt = tripO;   break;
    case DM_POLYTOPE_TRI_PRISM_TENSOR:   *Nt = 2; *target = ttriT;   *size = ttriS;   *cone = ttriC;   *ornt = ttriO;   break;
    case DM_POLYTOPE_QUAD_PRISM_TENSOR:  *Nt = 3; *target = tquadT;  *size = tquadS;  *cone = tquadC;  *ornt = tquadO;  break;
    case DM_POLYTOPE_PYRAMID:            *Nt = 5; *target = tpyrT;   *size = tpyrS;   *cone = tpyrC;   *ornt = tpyrO;   break;
    default: SETERRQ(PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "No refinement strategy for %s", DMPolytopeTypes[source]);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode DMPlexTransformInitialize_Regular(DMPlexTransform tr)
{
  PetscFunctionBegin;
  tr->ops->view    = DMPlexTransformView_Regular;
  tr->ops->setup   = DMPlexTransformSetUp_Regular;
  tr->ops->destroy = DMPlexTransformDestroy_Regular;
  tr->ops->celltransform = DMPlexTransformCellRefine_Regular;
  tr->ops->getsubcellorientation = DMPlexTransformGetSubcellOrientation_Regular;
  tr->ops->mapcoordinates = DMPlexTransformMapCoordinatesBarycenter_Internal;
  PetscFunctionReturn(0);
}

PETSC_EXTERN PetscErrorCode DMPlexTransformCreate_Regular(DMPlexTransform tr)
{
  DMPlexRefine_Regular *f;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(tr, DMPLEXTRANSFORM_CLASSID, 1);
  CHKERRQ(PetscNewLog(tr, &f));
  tr->data = f;

  CHKERRQ(DMPlexTransformInitialize_Regular(tr));
  PetscFunctionReturn(0);
}
