static char help[] = "Mesh Orientation Tutorial\n\n";

#include <petscdmplex.h>
#include <petscdmplextransform.h>

typedef struct {
  PetscBool genArr;        /* Generate all possible cell arrangements */
  PetscBool refArr;        /* Refine all possible cell arrangements */
  PetscBool printTable;    /* Print the CAyley table */
  PetscInt  orntBounds[2]; /* Bounds for the orientation check */
  PetscInt  numOrnt;       /* Number of specific orientations specified, or -1 for all orientations */
  PetscInt  ornts[48];     /* Specific orientations if specified */
  PetscInt  initOrnt;      /* Initial orientation for starting mesh */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt       n = 2;
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  options->genArr        = PETSC_FALSE;
  options->refArr        = PETSC_FALSE;
  options->printTable    = PETSC_FALSE;
  options->orntBounds[0] = PETSC_MIN_INT;
  options->orntBounds[1] = PETSC_MAX_INT;
  options->numOrnt       = -1;
  options->initOrnt      = 0;

  ierr = PetscOptionsBegin(comm, "", "Mesh Orientation Tutorials Options", "DMPLEX");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-gen_arrangements", "Flag for generating all arrangements of the cell", "ex11.c", options->genArr, &options->genArr, NULL));
  CHKERRQ(PetscOptionsBool("-ref_arrangements", "Flag for refining all arrangements of the cell", "ex11.c", options->refArr, &options->refArr, NULL));
  CHKERRQ(PetscOptionsBool("-print_table", "Print the Cayley table", "ex11.c", options->printTable, &options->printTable, NULL));
  CHKERRQ(PetscOptionsIntArray("-ornt_bounds", "Bounds for orientation checks", "ex11.c", options->orntBounds, &n, NULL));
  n    = 48;
  CHKERRQ(PetscOptionsIntArray("-ornts", "Specific orientations for checks", "ex11.c", options->ornts, &n, &flg));
  if (flg) {
    options->numOrnt = n;
    CHKERRQ(PetscSortInt(n, options->ornts));
  }
  CHKERRQ(PetscOptionsInt("-init_ornt", "Initial orientation for starting mesh", "ex11.c", options->initOrnt, &options->initOrnt, NULL));
  ierr = PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscBool ignoreOrnt(AppCtx *user, PetscInt o)
{
  PetscInt       loc;
  PetscErrorCode ierr;

  if (user->numOrnt < 0) return PETSC_FALSE;
  ierr = PetscFindInt(o, user->numOrnt, user->ornts, &loc);
  if (loc < 0 || ierr)   return PETSC_TRUE;
  return PETSC_FALSE;
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckCellVertices(DM dm, PetscInt cell, PetscInt o)
{
  DMPolytopeType  ct;
  const PetscInt *arrVerts;
  PetscInt       *closure = NULL;
  PetscInt        Ncl, cl, Nv, vStart, vEnd, v;
  MPI_Comm        comm;

  PetscFunctionBeginUser;
  CHKERRQ(PetscObjectGetComm((PetscObject) dm, &comm));
  CHKERRQ(DMPlexGetCellType(dm, cell, &ct));
  CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  CHKERRQ(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
  for (cl = 0, Nv = 0; cl < Ncl*2; cl += 2) {
    const PetscInt vertex = closure[cl];

    if (vertex < vStart || vertex >= vEnd) continue;
    closure[Nv++] = vertex;
  }
  PetscCheckFalse(Nv != DMPolytopeTypeGetNumVertices(ct),comm, PETSC_ERR_ARG_WRONG, "Cell %D has %D vertices != %D vertices in a %s", cell, Nv, DMPolytopeTypeGetNumVertices(ct), DMPolytopeTypes[ct]);
  arrVerts = DMPolytopeTypeGetVertexArrangment(ct, o);
  for (v = 0; v < Nv; ++v) {
    PetscCheckFalse(closure[v] != arrVerts[v]+vStart,comm, PETSC_ERR_ARG_WRONG, "Cell %D vertex[%D]: %D should be %D for arrangement %D", cell, v, closure[v], arrVerts[v]+vStart, o);
  }
  CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
  PetscFunctionReturn(0);
}

/* Transform cell with group operation o */
static PetscErrorCode ReorientCell(DM dm, PetscInt cell, PetscInt o, PetscBool swapCoords)
{
  DM              cdm;
  Vec             coordinates;
  PetscScalar    *coords, *ccoords = NULL;
  PetscInt       *closure = NULL;
  PetscInt        cdim, d, Nc, Ncl, cl, vStart, vEnd, Nv;

  PetscFunctionBegin;
  /* Change vertex coordinates so that it plots as we expect */
  CHKERRQ(DMGetCoordinateDM(dm, &cdm));
  CHKERRQ(DMGetCoordinateDim(dm, &cdim));
  CHKERRQ(DMGetCoordinatesLocal(dm, &coordinates));
  CHKERRQ(DMPlexVecGetClosure(cdm, NULL, coordinates, cell, &Nc, &ccoords));
  /* Reorient cone */
  CHKERRQ(DMPlexOrientPoint(dm, cell, o));
  /* Finish resetting coordinates */
  if (swapCoords) {
    CHKERRQ(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    CHKERRQ(VecGetArrayWrite(coordinates, &coords));
    CHKERRQ(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
    for (cl = 0, Nv = 0; cl < Ncl*2; cl += 2) {
      const PetscInt vertex = closure[cl];
      PetscScalar   *vcoords;

      if (vertex < vStart || vertex >= vEnd) continue;
      CHKERRQ(DMPlexPointLocalRef(cdm, vertex, coords, &vcoords));
      for (d = 0; d < cdim; ++d) vcoords[d] = ccoords[Nv*cdim + d];
      ++Nv;
    }
    CHKERRQ(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
    CHKERRQ(VecRestoreArrayWrite(coordinates, &coords));
  }
  CHKERRQ(DMPlexVecRestoreClosure(cdm, NULL, coordinates, cell, &Nc, &ccoords));
  PetscFunctionReturn(0);
}

static PetscErrorCode GenerateArrangments(DM dm, AppCtx *user)
{
  DM             odm;
  DMPolytopeType ct;
  PetscInt       No, o;
  const char    *name;

  PetscFunctionBeginUser;
  if (!user->genArr) PetscFunctionReturn(0);
  CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
  CHKERRQ(DMPlexGetCellType(dm, 0, &ct));
  No   = DMPolytopeTypeGetNumArrangments(ct)/2;
  for (o = PetscMax(-No, user->orntBounds[0]); o < PetscMin(No, user->orntBounds[1]); ++o) {
    if (ignoreOrnt(user, o)) continue;
    CHKERRQ(CreateMesh(PetscObjectComm((PetscObject) dm), user, &odm));
    CHKERRQ(ReorientCell(odm, 0, o, PETSC_TRUE));
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "%s orientation %D\n", name, o));
    CHKERRQ(DMViewFromOptions(odm, NULL, "-gen_dm_view"));
    CHKERRQ(CheckCellVertices(odm, 0, o));
    CHKERRQ(DMDestroy(&odm));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VerifyCayleyTable(DM dm, AppCtx *user)
{
  DM              dm1, dm2;
  DMPolytopeType  ct;
  const PetscInt *refcone, *cone;
  PetscInt        No, o1, o2, o3, o4;
  PetscBool       equal;
  const char     *name;

  PetscFunctionBeginUser;
  if (!user->genArr) PetscFunctionReturn(0);
  CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
  CHKERRQ(DMPlexGetCellType(dm, 0, &ct));
  CHKERRQ(DMPlexGetCone(dm, 0, &refcone));
  No   = DMPolytopeTypeGetNumArrangments(ct)/2;
  if (user->printTable) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Cayley Table for %s\n", DMPolytopeTypes[ct]));
  for (o1 = PetscMax(-No, user->orntBounds[0]); o1 < PetscMin(No, user->orntBounds[1]); ++o1) {
    for (o2 = PetscMax(-No, user->orntBounds[0]); o2 < PetscMin(No, user->orntBounds[1]); ++o2) {
      CHKERRQ(CreateMesh(PetscObjectComm((PetscObject) dm), user, &dm1));
      CHKERRQ(DMPlexOrientPoint(dm1, 0, o2));
      CHKERRQ(DMPlexCheckFaces(dm1, 0));
      CHKERRQ(DMPlexOrientPoint(dm1, 0, o1));
      CHKERRQ(DMPlexCheckFaces(dm1, 0));
      o3   = DMPolytopeTypeComposeOrientation(ct, o1, o2);
      /* First verification */
      CHKERRQ(CreateMesh(PetscObjectComm((PetscObject) dm), user, &dm2));
      CHKERRQ(DMPlexOrientPoint(dm2, 0, o3));
      CHKERRQ(DMPlexCheckFaces(dm2, 0));
      CHKERRQ(DMPlexEqual(dm1, dm2, &equal));
      if (!equal) {
        CHKERRQ(DMViewFromOptions(dm1, NULL, "-error_dm_view"));
        CHKERRQ(DMViewFromOptions(dm2, NULL, "-error_dm_view"));
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cayley table error for %s: %D * %D != %D", DMPolytopeTypes[ct], o1, o2, o3);
      }
      /* Second verification */
      CHKERRQ(DMPlexGetCone(dm1, 0, &cone));
      CHKERRQ(DMPolytopeGetOrientation(ct, refcone, cone, &o4));
      if (user->printTable) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%D, ", o4));
      PetscCheckFalse(o3 != o4,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cayley table error for %s: %D * %D = %D != %D", DMPolytopeTypes[ct], o1, o2, o3, o4);
      CHKERRQ(DMDestroy(&dm1));
      CHKERRQ(DMDestroy(&dm2));
    }
    if (user->printTable) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode VerifyInverse(DM dm, AppCtx *user)
{
  DM              dm1, dm2;
  DMPolytopeType  ct;
  const PetscInt *refcone, *cone;
  PetscInt        No, o, oi, o2;
  PetscBool       equal;
  const char     *name;

  PetscFunctionBeginUser;
  if (!user->genArr) PetscFunctionReturn(0);
  CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
  CHKERRQ(DMPlexGetCellType(dm, 0, &ct));
  CHKERRQ(DMPlexGetCone(dm, 0, &refcone));
  No   = DMPolytopeTypeGetNumArrangments(ct)/2;
  if (user->printTable) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Inverse table for %s\n", DMPolytopeTypes[ct]));
  for (o = PetscMax(-No, user->orntBounds[0]); o < PetscMin(No, user->orntBounds[1]); ++o) {
    if (ignoreOrnt(user, o)) continue;
    oi   = DMPolytopeTypeComposeOrientationInv(ct, 0, o);
    CHKERRQ(CreateMesh(PetscObjectComm((PetscObject) dm), user, &dm1));
    CHKERRQ(DMPlexOrientPoint(dm1, 0, o));
    CHKERRQ(DMPlexCheckFaces(dm1, 0));
    CHKERRQ(DMPlexOrientPoint(dm1, 0, oi));
    CHKERRQ(DMPlexCheckFaces(dm1, 0));
    /* First verification */
    CHKERRQ(CreateMesh(PetscObjectComm((PetscObject) dm), user, &dm2));
    CHKERRQ(DMPlexEqual(dm1, dm2, &equal));
    if (!equal) {
      CHKERRQ(DMViewFromOptions(dm1, NULL, "-error_dm_view"));
      CHKERRQ(DMViewFromOptions(dm2, NULL, "-error_dm_view"));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverse error for %s: %D * %D != 0", DMPolytopeTypes[ct], o, oi);
    }
    /* Second verification */
    CHKERRQ(DMPlexGetCone(dm1, 0, &cone));
    CHKERRQ(DMPolytopeGetOrientation(ct, refcone, cone, &o2));
    if (user->printTable) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%D, ", oi));
    PetscCheckFalse(o2 != 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverse error for %s: %D * %D = %D != 0", DMPolytopeTypes[ct], o, oi, o2);
    CHKERRQ(DMDestroy(&dm1));
    CHKERRQ(DMDestroy(&dm2));
  }
  if (user->printTable) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
  PetscFunctionReturn(0);
}

/* Suppose that point p has the same arrangement as o from canonical, compare the subcells to canonical subcells */
static PetscErrorCode CheckSubcells(DM dm, DM odm, PetscInt p, PetscInt o, AppCtx *user)
{
  DMPlexTransform tr, otr;
  DMPolytopeType  ct;
  DMPolytopeType *rct;
  const PetscInt *cone, *ornt, *ocone, *oornt;
  PetscInt       *rsize, *rcone, *rornt;
  PetscInt        Nct, n, oi, debug = 0;

  PetscFunctionBeginUser;
  CHKERRQ(DMPlexTransformCreate(PetscObjectComm((PetscObject) dm), &tr));
  CHKERRQ(DMPlexTransformSetDM(tr, dm));
  CHKERRQ(DMPlexTransformSetFromOptions(tr));
  CHKERRQ(DMPlexTransformSetUp(tr));

  CHKERRQ(DMPlexTransformCreate(PetscObjectComm((PetscObject) odm), &otr));
  CHKERRQ(DMPlexTransformSetDM(otr, odm));
  CHKERRQ(DMPlexTransformSetFromOptions(otr));
  CHKERRQ(DMPlexTransformSetUp(otr));

  CHKERRQ(DMPlexGetCellType(dm, p, &ct));
  CHKERRQ(DMPlexGetCone(dm, p, &cone));
  CHKERRQ(DMPlexGetConeOrientation(dm, p, &ornt));
  CHKERRQ(DMPlexGetCone(odm, p, &ocone));
  CHKERRQ(DMPlexGetConeOrientation(odm, p, &oornt));
  oi   = DMPolytopeTypeComposeOrientationInv(ct, 0, o);
  if (user->printTable) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "Orientation %D\n", oi));

  CHKERRQ(DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
  for (n = 0; n < Nct; ++n) {
    DMPolytopeType ctNew = rct[n];
    PetscInt       r, ro;

    if (debug) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "  Checking type %s\n", DMPolytopeTypes[ctNew]));
    for (r = 0; r < rsize[n]; ++r) {
      const PetscInt *qcone, *qornt, *oqcone, *oqornt;
      PetscInt        pNew, opNew, oo, pr, fo;
      PetscBool       restore = PETSC_TRUE;

      CHKERRQ(DMPlexTransformGetTargetPoint(tr, ct, ctNew, p, r, &pNew));
      CHKERRQ(DMPlexTransformGetCone(tr, pNew, &qcone, &qornt));
      if (debug) {
        PetscInt c;

        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "    Checking replica %D (%D)\n      Original Cone", r, pNew));
        for (c = 0; c < DMPolytopeTypeGetConeSize(ctNew); ++c) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " %D (%D)", qcone[c], qornt[c]));
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
      }
      for (ro = 0; ro < rsize[n]; ++ro) {
        PetscBool found;

        CHKERRQ(DMPlexTransformGetTargetPoint(otr, ct, ctNew, p, ro, &opNew));
        CHKERRQ(DMPlexTransformGetConeOriented(otr, opNew, o, &oqcone, &oqornt));
        CHKERRQ(DMPolytopeMatchOrientation(ctNew, oqcone, qcone, &oo, &found));
        if (found) break;
        CHKERRQ(DMPlexTransformRestoreCone(otr, pNew, &oqcone, &oqornt));
      }
      if (debug) {
        PetscInt c;

        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "    Checking transform replica %D (%D) (%D)\n      Transform Cone", ro, opNew, o));
        for (c = 0; c < DMPolytopeTypeGetConeSize(ctNew); ++c) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " %D (%D)", oqcone[c], oqornt[c]));
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "    Matched %D\n", oo));
      }
      if (ro == rsize[n]) {
        /* The tetrahedron has 3 pairs of opposing edges, and any pair can be connected by the interior segment */
        if (ct == DM_POLYTOPE_TETRAHEDRON) {
          /* The segment in a tetrahedron does not map into itself under the group action */
          if (ctNew == DM_POLYTOPE_SEGMENT) {restore = PETSC_FALSE; ro = r; oo = 0;}
          /* The last four interior faces do not map into themselves under the group action */
          if (r > 3 && ctNew == DM_POLYTOPE_TRIANGLE) {restore = PETSC_FALSE; ro = r; oo = 0;}
          /* The last four interior faces do not map into themselves under the group action */
          if (r > 3 && ctNew == DM_POLYTOPE_TETRAHEDRON) {restore = PETSC_FALSE; ro = r; oo = 0;}
        }
        PetscCheck(!restore,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to find matching %s %D orientation for cell orientation %D", DMPolytopeTypes[ctNew], r, o);
      }
      if (user->printTable) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "%D, %D, ", ro, oo));
      CHKERRQ(DMPlexTransformGetSubcellOrientation(tr, ct, p, oi, ctNew, r, 0, &pr, &fo));
      if (!user->printTable) {
        PetscCheckFalse(pr != ro,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Choose wrong replica %D != %D", pr, ro);
        PetscCheckFalse(fo != oo,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Choose wrong orientation %D != %D", fo, oo);
      }
      CHKERRQ(DMPlexTransformRestoreCone(tr, pNew, &qcone, &qornt));
      if (restore) CHKERRQ(DMPlexTransformRestoreCone(otr, pNew, &oqcone, &oqornt));
    }
    if (user->printTable) CHKERRQ(PetscPrintf(PETSC_COMM_SELF, "\n"));
  }
  CHKERRQ(DMPlexTransformDestroy(&tr));
  CHKERRQ(DMPlexTransformDestroy(&otr));
  PetscFunctionReturn(0);
}

static PetscErrorCode RefineArrangments(DM dm, AppCtx *user)
{
  DM             odm, rdm;
  DMPolytopeType ct;
  PetscInt       No, o;
  const char    *name;

  PetscFunctionBeginUser;
  if (!user->refArr) PetscFunctionReturn(0);
  CHKERRQ(PetscObjectGetName((PetscObject) dm, &name));
  CHKERRQ(DMPlexGetCellType(dm, 0, &ct));
  No   = DMPolytopeTypeGetNumArrangments(ct)/2;
  for (o = PetscMax(-No, user->orntBounds[0]); o < PetscMin(No, user->orntBounds[1]); ++o) {
    if (ignoreOrnt(user, o)) continue;
    CHKERRQ(CreateMesh(PetscObjectComm((PetscObject) dm), user, &odm));
    if (user->initOrnt) CHKERRQ(ReorientCell(odm, 0, user->initOrnt, PETSC_FALSE));
    CHKERRQ(ReorientCell(odm, 0, o, PETSC_TRUE));
    CHKERRQ(DMViewFromOptions(odm, NULL, "-orig_dm_view"));
    CHKERRQ(DMRefine(odm, MPI_COMM_NULL, &rdm));
    CHKERRQ(DMSetFromOptions(rdm));
    CHKERRQ(PetscPrintf(PetscObjectComm((PetscObject) dm), "%s orientation %D\n", name, o));
    CHKERRQ(DMViewFromOptions(rdm, NULL, "-ref_dm_view"));
    CHKERRQ(CheckSubcells(dm, odm, 0, o, user));
    CHKERRQ(DMDestroy(&odm));
    CHKERRQ(DMDestroy(&rdm));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(ProcessOptions(PETSC_COMM_WORLD, &user));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  if (user.initOrnt) {
    CHKERRQ(ReorientCell(dm, 0, user.initOrnt, PETSC_FALSE));
    CHKERRQ(DMViewFromOptions(dm, NULL, "-ornt_dm_view"));
  }
  CHKERRQ(GenerateArrangments(dm, &user));
  CHKERRQ(VerifyCayleyTable(dm, &user));
  CHKERRQ(VerifyInverse(dm, &user));
  CHKERRQ(RefineArrangments(dm, &user));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  testset:
    args: -dm_coord_space 0 -dm_plex_reference_cell_domain -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_tikzscale 0.5

    test:
      suffix: segment
      args: -dm_plex_cell segment \
            -dm_plex_view_numbers_depth 1,0 -dm_plex_view_colors_depth 1,0

    test:
      suffix: triangle
      args: -dm_plex_cell triangle \
            -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0

    test:
      suffix: quadrilateral
      args: -dm_plex_cell quadrilateral \
            -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0

    test:
      suffix: tensor_segment
      args: -dm_plex_cell tensor_quad \
            -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0

    test:
      suffix: tetrahedron
      args: -dm_plex_cell tetrahedron \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0

    test:
      suffix: hexahedron
      args: -dm_plex_cell hexahedron \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 0.3

    test:
      suffix: triangular_prism
      args: -dm_plex_cell triangular_prism \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0

    test:
      suffix: tensor_triangular_prism
      args: -dm_plex_cell tensor_triangular_prism \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0

    test:
      suffix: tensor_quadrilateral_prism
      args: -dm_plex_cell tensor_quadrilateral_prism \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0

    test:
      suffix: pyramid
      args: -dm_plex_cell pyramid \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0

  testset:
    args: -dm_coord_space 0 -dm_plex_reference_cell_domain -ref_arrangements -dm_plex_check_all \
          -ref_dm_view ::ascii_latex -dm_plex_view_tikzscale 1.0

    test:
      suffix: ref_segment
      args: -dm_plex_cell segment \
            -dm_plex_view_numbers_depth 1,0 -dm_plex_view_colors_depth 1,0

    test:
      suffix: ref_triangle
      args: -dm_plex_cell triangle \
            -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0

    test:
      suffix: ref_quadrilateral
      args: -dm_plex_cell quadrilateral \
            -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0

    test:
      suffix: ref_tensor_segment
      args: -dm_plex_cell tensor_quad \
            -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0

    test:
      suffix: ref_tetrahedron
      args: -dm_plex_cell tetrahedron \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0

    test:
      suffix: ref_hexahedron
      args: -dm_plex_cell hexahedron \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0

    test:
      suffix: ref_triangular_prism
      args: -dm_plex_cell triangular_prism \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0

    test:
      suffix: ref_tensor_triangular_prism
      args: -dm_plex_cell tensor_triangular_prism \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0

    test:
      suffix: ref_tensor_quadrilateral_prism
      args: -dm_plex_cell tensor_quadrilateral_prism \
            -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0

  # ToBox should recreate the coordinate space since the cell shape changes
  testset:
    args: -dm_coord_space 0 -dm_plex_transform_type refine_tobox -ref_arrangements -dm_plex_check_all

    test:
      suffix: tobox_triangle
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    test:
      suffix: tobox_tensor_segment
      args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_quad \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    test:
      suffix: tobox_tetrahedron
      args: -dm_plex_reference_cell_domain -dm_plex_cell tetrahedron \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

    test:
      suffix: tobox_triangular_prism
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangular_prism \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

    test:
      suffix: tobox_tensor_triangular_prism
      args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_triangular_prism \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

    test:
      suffix: tobox_tensor_quadrilateral_prism
      args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_quadrilateral_prism \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

  testset:
    args: -dm_coord_space 0 -dm_plex_transform_type refine_alfeld -ref_arrangements -dm_plex_check_all

    test:
      suffix: alfeld_triangle
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    test:
      suffix: alfeld_tetrahedron
      args: -dm_plex_reference_cell_domain -dm_plex_cell tetrahedron \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

  testset:
    args: -dm_plex_transform_type refine_sbr -ref_arrangements -dm_plex_check_all

    # This splits edge 1 of the triangle, and reflects about the added edge
    test:
      suffix: sbr_triangle_0
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_plex_transform_sbr_ref_cell 5 -ornts -2,0 \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    # This splits edge 0 of the triangle, and reflects about the added edge
    test:
      suffix: sbr_triangle_1
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_plex_transform_sbr_ref_cell 5 -init_ornt 1 -ornts -3,0 \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    # This splits edge 2 of the triangle, and reflects about the added edge
    test:
      suffix: sbr_triangle_2
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_plex_transform_sbr_ref_cell 5 -init_ornt 2 -ornts -1,0 \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    # This splits edges 1 and 2 of the triangle
    test:
      suffix: sbr_triangle_3
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_plex_transform_sbr_ref_cell 5,6 -ornts 0 \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    # This splits edges 1 and 0 of the triangle
    test:
      suffix: sbr_triangle_4
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_plex_transform_sbr_ref_cell 4,5 -ornts 0 \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    # This splits edges 0 and 1 of the triangle
    test:
      suffix: sbr_triangle_5
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_plex_transform_sbr_ref_cell 5,6 -init_ornt 1 -ornts 0 \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    # This splits edges 0 and 2 of the triangle
    test:
      suffix: sbr_triangle_6
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_plex_transform_sbr_ref_cell 4,5 -init_ornt 1 -ornts 0 \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    # This splits edges 2 and 0 of the triangle
    test:
      suffix: sbr_triangle_7
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_plex_transform_sbr_ref_cell 5,6 -init_ornt 2 -ornts 0 \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    # This splits edges 2 and 1 of the triangle
    test:
      suffix: sbr_triangle_8
      args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -dm_plex_transform_sbr_ref_cell 4,5 -init_ornt 2 -ornts 0 \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

  testset:
    args: -dm_plex_transform_type refine_boundary_layer -dm_plex_transform_bl_splits 2 -ref_arrangements -dm_plex_check_all

    test:
      suffix: bl_tensor_segment
      args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_quad \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

    # The subcell check is broken because at orientation 3, the internal triangles do not get properly permuted for the check
    test:
      suffix: bl_tensor_triangular_prism
      requires: TODO
      args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_triangular_prism \
            -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

TEST*/
