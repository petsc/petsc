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
  PetscInt  n = 2;
  PetscBool flg;

  PetscFunctionBeginUser;
  options->genArr        = PETSC_FALSE;
  options->refArr        = PETSC_FALSE;
  options->printTable    = PETSC_FALSE;
  options->orntBounds[0] = PETSC_MIN_INT;
  options->orntBounds[1] = PETSC_MAX_INT;
  options->numOrnt       = -1;
  options->initOrnt      = 0;

  PetscOptionsBegin(comm, "", "Mesh Orientation Tutorials Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-gen_arrangements", "Flag for generating all arrangements of the cell", "ex11.c", options->genArr, &options->genArr, NULL));
  PetscCall(PetscOptionsBool("-ref_arrangements", "Flag for refining all arrangements of the cell", "ex11.c", options->refArr, &options->refArr, NULL));
  PetscCall(PetscOptionsBool("-print_table", "Print the Cayley table", "ex11.c", options->printTable, &options->printTable, NULL));
  PetscCall(PetscOptionsIntArray("-ornt_bounds", "Bounds for orientation checks", "ex11.c", options->orntBounds, &n, NULL));
  n = 48;
  PetscCall(PetscOptionsIntArray("-ornts", "Specific orientations for checks", "ex11.c", options->ornts, &n, &flg));
  if (flg) {
    options->numOrnt = n;
    PetscCall(PetscSortInt(n, options->ornts));
  }
  PetscCall(PetscOptionsInt("-init_ornt", "Initial orientation for starting mesh", "ex11.c", options->initOrnt, &options->initOrnt, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(0);
}

static PetscBool ignoreOrnt(AppCtx *user, PetscInt o)
{
  PetscInt       loc;
  PetscErrorCode ierr;

  if (user->numOrnt < 0) return PETSC_FALSE;
  ierr = PetscFindInt(o, user->numOrnt, user->ornts, &loc);
  if (loc < 0 || ierr) return PETSC_TRUE;
  return PETSC_FALSE;
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
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
  PetscCall(PetscObjectGetComm((PetscObject)dm, &comm));
  PetscCall(DMPlexGetCellType(dm, cell, &ct));
  PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
  PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
  for (cl = 0, Nv = 0; cl < Ncl * 2; cl += 2) {
    const PetscInt vertex = closure[cl];

    if (vertex < vStart || vertex >= vEnd) continue;
    closure[Nv++] = vertex;
  }
  PetscCheck(Nv == DMPolytopeTypeGetNumVertices(ct), comm, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " has %" PetscInt_FMT " vertices != %" PetscInt_FMT " vertices in a %s", cell, Nv, DMPolytopeTypeGetNumVertices(ct), DMPolytopeTypes[ct]);
  arrVerts = DMPolytopeTypeGetVertexArrangment(ct, o);
  for (v = 0; v < Nv; ++v) {
    PetscCheck(closure[v] == arrVerts[v] + vStart, comm, PETSC_ERR_ARG_WRONG, "Cell %" PetscInt_FMT " vertex[%" PetscInt_FMT "]: %" PetscInt_FMT " should be %" PetscInt_FMT " for arrangement %" PetscInt_FMT, cell, v, closure[v], arrVerts[v] + vStart, o);
  }
  PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
  PetscFunctionReturn(0);
}

/* Transform cell with group operation o */
static PetscErrorCode ReorientCell(DM dm, PetscInt cell, PetscInt o, PetscBool swapCoords)
{
  DM           cdm;
  Vec          coordinates;
  PetscScalar *coords, *ccoords = NULL;
  PetscInt    *closure = NULL;
  PetscInt     cdim, d, Nc, Ncl, cl, vStart, vEnd, Nv;

  PetscFunctionBegin;
  /* Change vertex coordinates so that it plots as we expect */
  PetscCall(DMGetCoordinateDM(dm, &cdm));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  PetscCall(DMGetCoordinatesLocal(dm, &coordinates));
  PetscCall(DMPlexVecGetClosure(cdm, NULL, coordinates, cell, &Nc, &ccoords));
  /* Reorient cone */
  PetscCall(DMPlexOrientPoint(dm, cell, o));
  /* Finish resetting coordinates */
  if (swapCoords) {
    PetscCall(DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd));
    PetscCall(VecGetArrayWrite(coordinates, &coords));
    PetscCall(DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
    for (cl = 0, Nv = 0; cl < Ncl * 2; cl += 2) {
      const PetscInt vertex = closure[cl];
      PetscScalar   *vcoords;

      if (vertex < vStart || vertex >= vEnd) continue;
      PetscCall(DMPlexPointLocalRef(cdm, vertex, coords, &vcoords));
      for (d = 0; d < cdim; ++d) vcoords[d] = ccoords[Nv * cdim + d];
      ++Nv;
    }
    PetscCall(DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure));
    PetscCall(VecRestoreArrayWrite(coordinates, &coords));
  }
  PetscCall(DMPlexVecRestoreClosure(cdm, NULL, coordinates, cell, &Nc, &ccoords));
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
  PetscCall(PetscObjectGetName((PetscObject)dm, &name));
  PetscCall(DMPlexGetCellType(dm, 0, &ct));
  No = DMPolytopeTypeGetNumArrangments(ct) / 2;
  for (o = PetscMax(-No, user->orntBounds[0]); o < PetscMin(No, user->orntBounds[1]); ++o) {
    if (ignoreOrnt(user, o)) continue;
    PetscCall(CreateMesh(PetscObjectComm((PetscObject)dm), user, &odm));
    PetscCall(ReorientCell(odm, 0, o, PETSC_TRUE));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "%s orientation %" PetscInt_FMT "\n", name, o));
    PetscCall(DMViewFromOptions(odm, NULL, "-gen_dm_view"));
    PetscCall(CheckCellVertices(odm, 0, o));
    PetscCall(DMDestroy(&odm));
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
  PetscCall(PetscObjectGetName((PetscObject)dm, &name));
  PetscCall(DMPlexGetCellType(dm, 0, &ct));
  PetscCall(DMPlexGetCone(dm, 0, &refcone));
  No = DMPolytopeTypeGetNumArrangments(ct) / 2;
  if (user->printTable) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Cayley Table for %s\n", DMPolytopeTypes[ct]));
  for (o1 = PetscMax(-No, user->orntBounds[0]); o1 < PetscMin(No, user->orntBounds[1]); ++o1) {
    for (o2 = PetscMax(-No, user->orntBounds[0]); o2 < PetscMin(No, user->orntBounds[1]); ++o2) {
      PetscCall(CreateMesh(PetscObjectComm((PetscObject)dm), user, &dm1));
      PetscCall(DMPlexOrientPoint(dm1, 0, o2));
      PetscCall(DMPlexCheckFaces(dm1, 0));
      PetscCall(DMPlexOrientPoint(dm1, 0, o1));
      PetscCall(DMPlexCheckFaces(dm1, 0));
      o3 = DMPolytopeTypeComposeOrientation(ct, o1, o2);
      /* First verification */
      PetscCall(CreateMesh(PetscObjectComm((PetscObject)dm), user, &dm2));
      PetscCall(DMPlexOrientPoint(dm2, 0, o3));
      PetscCall(DMPlexCheckFaces(dm2, 0));
      PetscCall(DMPlexEqual(dm1, dm2, &equal));
      if (!equal) {
        PetscCall(DMViewFromOptions(dm1, NULL, "-error_dm_view"));
        PetscCall(DMViewFromOptions(dm2, NULL, "-error_dm_view"));
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cayley table error for %s: %" PetscInt_FMT " * %" PetscInt_FMT " != %" PetscInt_FMT, DMPolytopeTypes[ct], o1, o2, o3);
      }
      /* Second verification */
      PetscCall(DMPlexGetCone(dm1, 0, &cone));
      PetscCall(DMPolytopeGetOrientation(ct, refcone, cone, &o4));
      if (user->printTable) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT ", ", o4));
      PetscCheck(o3 == o4, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cayley table error for %s: %" PetscInt_FMT " * %" PetscInt_FMT " = %" PetscInt_FMT " != %" PetscInt_FMT, DMPolytopeTypes[ct], o1, o2, o3, o4);
      PetscCall(DMDestroy(&dm1));
      PetscCall(DMDestroy(&dm2));
    }
    if (user->printTable) PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
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
  PetscCall(PetscObjectGetName((PetscObject)dm, &name));
  PetscCall(DMPlexGetCellType(dm, 0, &ct));
  PetscCall(DMPlexGetCone(dm, 0, &refcone));
  No = DMPolytopeTypeGetNumArrangments(ct) / 2;
  if (user->printTable) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Inverse table for %s\n", DMPolytopeTypes[ct]));
  for (o = PetscMax(-No, user->orntBounds[0]); o < PetscMin(No, user->orntBounds[1]); ++o) {
    if (ignoreOrnt(user, o)) continue;
    oi = DMPolytopeTypeComposeOrientationInv(ct, 0, o);
    PetscCall(CreateMesh(PetscObjectComm((PetscObject)dm), user, &dm1));
    PetscCall(DMPlexOrientPoint(dm1, 0, o));
    PetscCall(DMPlexCheckFaces(dm1, 0));
    PetscCall(DMPlexOrientPoint(dm1, 0, oi));
    PetscCall(DMPlexCheckFaces(dm1, 0));
    /* First verification */
    PetscCall(CreateMesh(PetscObjectComm((PetscObject)dm), user, &dm2));
    PetscCall(DMPlexEqual(dm1, dm2, &equal));
    if (!equal) {
      PetscCall(DMViewFromOptions(dm1, NULL, "-error_dm_view"));
      PetscCall(DMViewFromOptions(dm2, NULL, "-error_dm_view"));
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverse error for %s: %" PetscInt_FMT " * %" PetscInt_FMT " != 0", DMPolytopeTypes[ct], o, oi);
    }
    /* Second verification */
    PetscCall(DMPlexGetCone(dm1, 0, &cone));
    PetscCall(DMPolytopeGetOrientation(ct, refcone, cone, &o2));
    if (user->printTable) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT ", ", oi));
    PetscCheck(o2 == 0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverse error for %s: %" PetscInt_FMT " * %" PetscInt_FMT " = %" PetscInt_FMT " != 0", DMPolytopeTypes[ct], o, oi, o2);
    PetscCall(DMDestroy(&dm1));
    PetscCall(DMDestroy(&dm2));
  }
  if (user->printTable) PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
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
  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), &tr));
  PetscCall(DMPlexTransformSetDM(tr, dm));
  PetscCall(DMPlexTransformSetFromOptions(tr));
  PetscCall(DMPlexTransformSetUp(tr));

  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)odm), &otr));
  PetscCall(DMPlexTransformSetDM(otr, odm));
  PetscCall(DMPlexTransformSetFromOptions(otr));
  PetscCall(DMPlexTransformSetUp(otr));

  PetscCall(DMPlexGetCellType(dm, p, &ct));
  PetscCall(DMPlexGetCone(dm, p, &cone));
  PetscCall(DMPlexGetConeOrientation(dm, p, &ornt));
  PetscCall(DMPlexGetCone(odm, p, &ocone));
  PetscCall(DMPlexGetConeOrientation(odm, p, &oornt));
  oi = DMPolytopeTypeComposeOrientationInv(ct, 0, o);
  if (user->printTable) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Orientation %" PetscInt_FMT "\n", oi));

  PetscCall(DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt));
  for (n = 0; n < Nct; ++n) {
    DMPolytopeType ctNew = rct[n];
    PetscInt       r, ro;

    if (debug) PetscCall(PetscPrintf(PETSC_COMM_SELF, "  Checking type %s\n", DMPolytopeTypes[ctNew]));
    for (r = 0; r < rsize[n]; ++r) {
      const PetscInt *qcone, *qornt, *oqcone, *oqornt;
      PetscInt        pNew, opNew, oo, pr, fo;
      PetscBool       restore = PETSC_TRUE;

      PetscCall(DMPlexTransformGetTargetPoint(tr, ct, ctNew, p, r, &pNew));
      PetscCall(DMPlexTransformGetCone(tr, pNew, &qcone, &qornt));
      if (debug) {
        PetscInt c;

        PetscCall(PetscPrintf(PETSC_COMM_SELF, "    Checking replica %" PetscInt_FMT " (%" PetscInt_FMT ")\n      Original Cone", r, pNew));
        for (c = 0; c < DMPolytopeTypeGetConeSize(ctNew); ++c) PetscCall(PetscPrintf(PETSC_COMM_SELF, " %" PetscInt_FMT " (%" PetscInt_FMT ")", qcone[c], qornt[c]));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
      }
      for (ro = 0; ro < rsize[n]; ++ro) {
        PetscBool found;

        PetscCall(DMPlexTransformGetTargetPoint(otr, ct, ctNew, p, ro, &opNew));
        PetscCall(DMPlexTransformGetConeOriented(otr, opNew, o, &oqcone, &oqornt));
        PetscCall(DMPolytopeMatchOrientation(ctNew, oqcone, qcone, &oo, &found));
        if (found) break;
        PetscCall(DMPlexTransformRestoreCone(otr, pNew, &oqcone, &oqornt));
      }
      if (debug) {
        PetscInt c;

        PetscCall(PetscPrintf(PETSC_COMM_SELF, "    Checking transform replica %" PetscInt_FMT " (%" PetscInt_FMT ") (%" PetscInt_FMT ")\n      Transform Cone", ro, opNew, o));
        for (c = 0; c < DMPolytopeTypeGetConeSize(ctNew); ++c) PetscCall(PetscPrintf(PETSC_COMM_SELF, " %" PetscInt_FMT " (%" PetscInt_FMT ")", oqcone[c], oqornt[c]));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
        PetscCall(PetscPrintf(PETSC_COMM_SELF, "    Matched %" PetscInt_FMT "\n", oo));
      }
      if (ro == rsize[n]) {
        /* The tetrahedron has 3 pairs of opposing edges, and any pair can be connected by the interior segment */
        if (ct == DM_POLYTOPE_TETRAHEDRON) {
          /* The segment in a tetrahedron does not map into itself under the group action */
          if (ctNew == DM_POLYTOPE_SEGMENT) {
            restore = PETSC_FALSE;
            ro      = r;
            oo      = 0;
          }
          /* The last four interior faces do not map into themselves under the group action */
          if (r > 3 && ctNew == DM_POLYTOPE_TRIANGLE) {
            restore = PETSC_FALSE;
            ro      = r;
            oo      = 0;
          }
          /* The last four interior faces do not map into themselves under the group action */
          if (r > 3 && ctNew == DM_POLYTOPE_TETRAHEDRON) {
            restore = PETSC_FALSE;
            ro      = r;
            oo      = 0;
          }
        }
        PetscCheck(!restore, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to find matching %s %" PetscInt_FMT " orientation for cell orientation %" PetscInt_FMT, DMPolytopeTypes[ctNew], r, o);
      }
      if (user->printTable) PetscCall(PetscPrintf(PETSC_COMM_SELF, "%" PetscInt_FMT ", %" PetscInt_FMT ", ", ro, oo));
      PetscCall(DMPlexTransformGetSubcellOrientation(tr, ct, p, oi, ctNew, r, 0, &pr, &fo));
      if (!user->printTable) {
        PetscCheck(pr == ro, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Choose wrong replica %" PetscInt_FMT " != %" PetscInt_FMT, pr, ro);
        PetscCheck(fo == oo, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Choose wrong orientation %" PetscInt_FMT " != %" PetscInt_FMT, fo, oo);
      }
      PetscCall(DMPlexTransformRestoreCone(tr, pNew, &qcone, &qornt));
      if (restore) PetscCall(DMPlexTransformRestoreCone(otr, pNew, &oqcone, &oqornt));
    }
    if (user->printTable) PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));
  }
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(DMPlexTransformDestroy(&otr));
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
  PetscCall(PetscObjectGetName((PetscObject)dm, &name));
  PetscCall(DMPlexGetCellType(dm, 0, &ct));
  No = DMPolytopeTypeGetNumArrangments(ct) / 2;
  for (o = PetscMax(-No, user->orntBounds[0]); o < PetscMin(No, user->orntBounds[1]); ++o) {
    if (ignoreOrnt(user, o)) continue;
    PetscCall(CreateMesh(PetscObjectComm((PetscObject)dm), user, &odm));
    if (user->initOrnt) PetscCall(ReorientCell(odm, 0, user->initOrnt, PETSC_FALSE));
    PetscCall(ReorientCell(odm, 0, o, PETSC_TRUE));
    PetscCall(DMViewFromOptions(odm, NULL, "-orig_dm_view"));
    PetscCall(DMRefine(odm, MPI_COMM_NULL, &rdm));
    PetscCall(DMSetFromOptions(rdm));
    PetscCall(PetscPrintf(PetscObjectComm((PetscObject)dm), "%s orientation %" PetscInt_FMT "\n", name, o));
    PetscCall(DMViewFromOptions(rdm, NULL, "-ref_dm_view"));
    PetscCall(CheckSubcells(dm, odm, 0, o, user));
    PetscCall(DMDestroy(&odm));
    PetscCall(DMDestroy(&rdm));
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM     dm;
  AppCtx user;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  if (user.initOrnt) {
    PetscCall(ReorientCell(dm, 0, user.initOrnt, PETSC_FALSE));
    PetscCall(DMViewFromOptions(dm, NULL, "-ornt_dm_view"));
  }
  PetscCall(GenerateArrangments(dm, &user));
  PetscCall(VerifyCayleyTable(dm, &user));
  PetscCall(VerifyInverse(dm, &user));
  PetscCall(RefineArrangments(dm, &user));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
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
