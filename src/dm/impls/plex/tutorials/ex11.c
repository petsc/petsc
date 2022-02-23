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
  ierr = PetscOptionsBool("-gen_arrangements", "Flag for generating all arrangements of the cell", "ex11.c", options->genArr, &options->genArr, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-ref_arrangements", "Flag for refining all arrangements of the cell", "ex11.c", options->refArr, &options->refArr, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsBool("-print_table", "Print the Cayley table", "ex11.c", options->printTable, &options->printTable, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsIntArray("-ornt_bounds", "Bounds for orientation checks", "ex11.c", options->orntBounds, &n, NULL);CHKERRQ(ierr);
  n    = 48;
  ierr = PetscOptionsIntArray("-ornts", "Specific orientations for checks", "ex11.c", options->ornts, &n, &flg);CHKERRQ(ierr);
  if (flg) {
    options->numOrnt = n;
    ierr = PetscSortInt(n, options->ornts);CHKERRQ(ierr);
  }
  ierr = PetscOptionsInt("-init_ornt", "Initial orientation for starting mesh", "ex11.c", options->initOrnt, &options->initOrnt, NULL);CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode CheckCellVertices(DM dm, PetscInt cell, PetscInt o)
{
  DMPolytopeType  ct;
  const PetscInt *arrVerts;
  PetscInt       *closure = NULL;
  PetscInt        Ncl, cl, Nv, vStart, vEnd, v;
  MPI_Comm        comm;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = PetscObjectGetComm((PetscObject) dm, &comm);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, cell, &ct);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
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
  ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBegin;
  /* Change vertex coordinates so that it plots as we expect */
  ierr = DMGetCoordinateDM(dm, &cdm);CHKERRQ(ierr);
  ierr = DMGetCoordinateDim(dm, &cdim);CHKERRQ(ierr);
  ierr = DMGetCoordinatesLocal(dm, &coordinates);CHKERRQ(ierr);
  ierr = DMPlexVecGetClosure(cdm, NULL, coordinates, cell, &Nc, &ccoords);CHKERRQ(ierr);
  /* Reorient cone */
  ierr = DMPlexOrientPoint(dm, cell, o);CHKERRQ(ierr);
  /* Finish resetting coordinates */
  if (swapCoords) {
    ierr = DMPlexGetDepthStratum(dm, 0, &vStart, &vEnd);CHKERRQ(ierr);
    ierr = VecGetArrayWrite(coordinates, &coords);CHKERRQ(ierr);
    ierr = DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
    for (cl = 0, Nv = 0; cl < Ncl*2; cl += 2) {
      const PetscInt vertex = closure[cl];
      PetscScalar   *vcoords;

      if (vertex < vStart || vertex >= vEnd) continue;
      ierr = DMPlexPointLocalRef(cdm, vertex, coords, &vcoords);CHKERRQ(ierr);
      for (d = 0; d < cdim; ++d) vcoords[d] = ccoords[Nv*cdim + d];
      ++Nv;
    }
    ierr = DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE, &Ncl, &closure);CHKERRQ(ierr);
    ierr = VecRestoreArrayWrite(coordinates, &coords);CHKERRQ(ierr);
  }
  ierr = DMPlexVecRestoreClosure(cdm, NULL, coordinates, cell, &Nc, &ccoords);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode GenerateArrangments(DM dm, AppCtx *user)
{
  DM             odm;
  DMPolytopeType ct;
  PetscInt       No, o;
  const char    *name;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (!user->genArr) PetscFunctionReturn(0);
  ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, 0, &ct);CHKERRQ(ierr);
  No   = DMPolytopeTypeGetNumArrangments(ct)/2;
  for (o = PetscMax(-No, user->orntBounds[0]); o < PetscMin(No, user->orntBounds[1]); ++o) {
    if (ignoreOrnt(user, o)) continue;
    ierr = CreateMesh(PetscObjectComm((PetscObject) dm), user, &odm);CHKERRQ(ierr);
    ierr = ReorientCell(odm, 0, o, PETSC_TRUE);CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "%s orientation %D\n", name, o);CHKERRQ(ierr);
    ierr = DMViewFromOptions(odm, NULL, "-gen_dm_view");CHKERRQ(ierr);
    ierr = CheckCellVertices(odm, 0, o);CHKERRQ(ierr);
    ierr = DMDestroy(&odm);CHKERRQ(ierr);
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
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  if (!user->genArr) PetscFunctionReturn(0);
  ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, 0, &ct);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, 0, &refcone);CHKERRQ(ierr);
  No   = DMPolytopeTypeGetNumArrangments(ct)/2;
  if (user->printTable) {ierr = PetscPrintf(PETSC_COMM_SELF, "Cayley Table for %s\n", DMPolytopeTypes[ct]);CHKERRQ(ierr);}
  for (o1 = PetscMax(-No, user->orntBounds[0]); o1 < PetscMin(No, user->orntBounds[1]); ++o1) {
    for (o2 = PetscMax(-No, user->orntBounds[0]); o2 < PetscMin(No, user->orntBounds[1]); ++o2) {
      ierr = CreateMesh(PetscObjectComm((PetscObject) dm), user, &dm1);CHKERRQ(ierr);
      ierr = DMPlexOrientPoint(dm1, 0, o2);CHKERRQ(ierr);
      ierr = DMPlexCheckFaces(dm1, 0);CHKERRQ(ierr);
      ierr = DMPlexOrientPoint(dm1, 0, o1);CHKERRQ(ierr);
      ierr = DMPlexCheckFaces(dm1, 0);CHKERRQ(ierr);
      o3   = DMPolytopeTypeComposeOrientation(ct, o1, o2);CHKERRQ(ierr);
      /* First verification */
      ierr = CreateMesh(PetscObjectComm((PetscObject) dm), user, &dm2);CHKERRQ(ierr);
      ierr = DMPlexOrientPoint(dm2, 0, o3);CHKERRQ(ierr);
      ierr = DMPlexCheckFaces(dm2, 0);CHKERRQ(ierr);
      ierr = DMPlexEqual(dm1, dm2, &equal);CHKERRQ(ierr);
      if (!equal) {
        ierr = DMViewFromOptions(dm1, NULL, "-error_dm_view");CHKERRQ(ierr);
        ierr = DMViewFromOptions(dm2, NULL, "-error_dm_view");CHKERRQ(ierr);
        SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cayley table error for %s: %D * %D != %D", DMPolytopeTypes[ct], o1, o2, o3);
      }
      /* Second verification */
      ierr = DMPlexGetCone(dm1, 0, &cone);CHKERRQ(ierr);
      ierr = DMPolytopeGetOrientation(ct, refcone, cone, &o4);CHKERRQ(ierr);
      if (user->printTable) {ierr = PetscPrintf(PETSC_COMM_SELF, "%D, ", o4);CHKERRQ(ierr);}
      PetscCheckFalse(o3 != o4,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Cayley table error for %s: %D * %D = %D != %D", DMPolytopeTypes[ct], o1, o2, o3, o4);
      ierr = DMDestroy(&dm1);CHKERRQ(ierr);
      ierr = DMDestroy(&dm2);CHKERRQ(ierr);
    }
    if (user->printTable) {ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);}
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
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  if (!user->genArr) PetscFunctionReturn(0);
  ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, 0, &ct);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, 0, &refcone);CHKERRQ(ierr);
  No   = DMPolytopeTypeGetNumArrangments(ct)/2;
  if (user->printTable) {ierr = PetscPrintf(PETSC_COMM_SELF, "Inverse table for %s\n", DMPolytopeTypes[ct]);CHKERRQ(ierr);}
  for (o = PetscMax(-No, user->orntBounds[0]); o < PetscMin(No, user->orntBounds[1]); ++o) {
    if (ignoreOrnt(user, o)) continue;
    oi   = DMPolytopeTypeComposeOrientationInv(ct, 0, o);CHKERRQ(ierr);
    ierr = CreateMesh(PetscObjectComm((PetscObject) dm), user, &dm1);CHKERRQ(ierr);
    ierr = DMPlexOrientPoint(dm1, 0, o);CHKERRQ(ierr);
    ierr = DMPlexCheckFaces(dm1, 0);CHKERRQ(ierr);
    ierr = DMPlexOrientPoint(dm1, 0, oi);CHKERRQ(ierr);
    ierr = DMPlexCheckFaces(dm1, 0);CHKERRQ(ierr);
    /* First verification */
    ierr = CreateMesh(PetscObjectComm((PetscObject) dm), user, &dm2);CHKERRQ(ierr);
    ierr = DMPlexEqual(dm1, dm2, &equal);CHKERRQ(ierr);
    if (!equal) {
      ierr = DMViewFromOptions(dm1, NULL, "-error_dm_view");CHKERRQ(ierr);
      ierr = DMViewFromOptions(dm2, NULL, "-error_dm_view");CHKERRQ(ierr);
      SETERRQ(PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverse error for %s: %D * %D != 0", DMPolytopeTypes[ct], o, oi);
    }
    /* Second verification */
    ierr = DMPlexGetCone(dm1, 0, &cone);CHKERRQ(ierr);
    ierr = DMPolytopeGetOrientation(ct, refcone, cone, &o2);CHKERRQ(ierr);
    if (user->printTable) {ierr = PetscPrintf(PETSC_COMM_SELF, "%D, ", oi);CHKERRQ(ierr);}
    PetscCheckFalse(o2 != 0,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Inverse error for %s: %D * %D = %D != 0", DMPolytopeTypes[ct], o, oi, o2);
    ierr = DMDestroy(&dm1);CHKERRQ(ierr);
    ierr = DMDestroy(&dm2);CHKERRQ(ierr);
  }
  if (user->printTable) {ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);}
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
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr = DMPlexTransformCreate(PetscObjectComm((PetscObject) dm), &tr);CHKERRQ(ierr);
  ierr = DMPlexTransformSetDM(tr, dm);CHKERRQ(ierr);
  ierr = DMPlexTransformSetFromOptions(tr);CHKERRQ(ierr);
  ierr = DMPlexTransformSetUp(tr);CHKERRQ(ierr);

  ierr = DMPlexTransformCreate(PetscObjectComm((PetscObject) odm), &otr);CHKERRQ(ierr);
  ierr = DMPlexTransformSetDM(otr, odm);CHKERRQ(ierr);
  ierr = DMPlexTransformSetFromOptions(otr);CHKERRQ(ierr);
  ierr = DMPlexTransformSetUp(otr);CHKERRQ(ierr);

  ierr = DMPlexGetCellType(dm, p, &ct);CHKERRQ(ierr);
  ierr = DMPlexGetCone(dm, p, &cone);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(dm, p, &ornt);CHKERRQ(ierr);
  ierr = DMPlexGetCone(odm, p, &ocone);CHKERRQ(ierr);
  ierr = DMPlexGetConeOrientation(odm, p, &oornt);CHKERRQ(ierr);
  oi   = DMPolytopeTypeComposeOrientationInv(ct, 0, o);CHKERRQ(ierr);
  if (user->printTable) {ierr = PetscPrintf(PETSC_COMM_SELF, "Orientation %D\n", oi);CHKERRQ(ierr);}

  ierr = DMPlexTransformCellTransform(tr, ct, p, NULL, &Nct, &rct, &rsize, &rcone, &rornt);CHKERRQ(ierr);
  for (n = 0; n < Nct; ++n) {
    DMPolytopeType ctNew = rct[n];
    PetscInt       r, ro;

    if (debug) {ierr = PetscPrintf(PETSC_COMM_SELF, "  Checking type %s\n", DMPolytopeTypes[ctNew]);CHKERRQ(ierr);}
    for (r = 0; r < rsize[n]; ++r) {
      const PetscInt *qcone, *qornt, *oqcone, *oqornt;
      PetscInt        pNew, opNew, oo, pr, fo;
      PetscBool       restore = PETSC_TRUE;

      ierr = DMPlexTransformGetTargetPoint(tr, ct, ctNew, p, r, &pNew);CHKERRQ(ierr);
      ierr = DMPlexTransformGetCone(tr, pNew, &qcone, &qornt);CHKERRQ(ierr);
      if (debug) {
        PetscInt c;

        ierr = PetscPrintf(PETSC_COMM_SELF, "    Checking replica %D (%D)\n      Original Cone", r, pNew);CHKERRQ(ierr);
        for (c = 0; c < DMPolytopeTypeGetConeSize(ctNew); ++c) {ierr = PetscPrintf(PETSC_COMM_SELF, " %D (%D)", qcone[c], qornt[c]);CHKERRQ(ierr);}
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
      }
      for (ro = 0; ro < rsize[n]; ++ro) {
        PetscBool found;

        ierr = DMPlexTransformGetTargetPoint(otr, ct, ctNew, p, ro, &opNew);CHKERRQ(ierr);
        ierr = DMPlexTransformGetConeOriented(otr, opNew, o, &oqcone, &oqornt);CHKERRQ(ierr);
        ierr = DMPolytopeMatchOrientation(ctNew, oqcone, qcone, &oo, &found);CHKERRQ(ierr);
        if (found) break;
        ierr = DMPlexTransformRestoreCone(otr, pNew, &oqcone, &oqornt);CHKERRQ(ierr);
      }
      if (debug) {
        PetscInt c;

        ierr = PetscPrintf(PETSC_COMM_SELF, "    Checking transform replica %D (%D) (%D)\n      Transform Cone", ro, opNew, o);CHKERRQ(ierr);
        for (c = 0; c < DMPolytopeTypeGetConeSize(ctNew); ++c) {ierr = PetscPrintf(PETSC_COMM_SELF, " %D (%D)", oqcone[c], oqornt[c]);CHKERRQ(ierr);}
        ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_SELF, "    Matched %D\n", oo);CHKERRQ(ierr);
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
        PetscCheckFalse(restore,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Unable to find matching %s %D orientation for cell orientation %D", DMPolytopeTypes[ctNew], r, o);
      }
      if (user->printTable) {ierr = PetscPrintf(PETSC_COMM_SELF, "%D, %D, ", ro, oo);CHKERRQ(ierr);}
      ierr = DMPlexTransformGetSubcellOrientation(tr, ct, p, oi, ctNew, r, 0, &pr, &fo);CHKERRQ(ierr);
      if (!user->printTable) {
        PetscCheckFalse(pr != ro,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Choose wrong replica %D != %D", pr, ro);
        PetscCheckFalse(fo != oo,PETSC_COMM_SELF, PETSC_ERR_PLIB, "Choose wrong orientation %D != %D", fo, oo);
      }
      ierr = DMPlexTransformRestoreCone(tr, pNew, &qcone, &qornt);CHKERRQ(ierr);
      if (restore) {ierr = DMPlexTransformRestoreCone(otr, pNew, &oqcone, &oqornt);CHKERRQ(ierr);}
    }
    if (user->printTable) {ierr = PetscPrintf(PETSC_COMM_SELF, "\n");CHKERRQ(ierr);}
  }
  ierr = DMPlexTransformDestroy(&tr);CHKERRQ(ierr);
  ierr = DMPlexTransformDestroy(&otr);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode RefineArrangments(DM dm, AppCtx *user)
{
  DM             odm, rdm;
  DMPolytopeType ct;
  PetscInt       No, o;
  const char    *name;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  if (!user->refArr) PetscFunctionReturn(0);
  ierr = PetscObjectGetName((PetscObject) dm, &name);CHKERRQ(ierr);
  ierr = DMPlexGetCellType(dm, 0, &ct);CHKERRQ(ierr);
  No   = DMPolytopeTypeGetNumArrangments(ct)/2;
  for (o = PetscMax(-No, user->orntBounds[0]); o < PetscMin(No, user->orntBounds[1]); ++o) {
    if (ignoreOrnt(user, o)) continue;
    ierr = CreateMesh(PetscObjectComm((PetscObject) dm), user, &odm);CHKERRQ(ierr);
    if (user->initOrnt) {ierr = ReorientCell(odm, 0, user->initOrnt, PETSC_FALSE);CHKERRQ(ierr);}
    ierr = ReorientCell(odm, 0, o, PETSC_TRUE);CHKERRQ(ierr);
    ierr = DMViewFromOptions(odm, NULL, "-orig_dm_view");CHKERRQ(ierr);
    ierr = DMRefine(odm, MPI_COMM_NULL, &rdm);CHKERRQ(ierr);
    ierr = DMSetFromOptions(rdm);CHKERRQ(ierr);
    ierr = PetscPrintf(PetscObjectComm((PetscObject) dm), "%s orientation %D\n", name, o);CHKERRQ(ierr);
    ierr = DMViewFromOptions(rdm, NULL, "-ref_dm_view");CHKERRQ(ierr);
    ierr = CheckSubcells(dm, odm, 0, o, user);CHKERRQ(ierr);
    ierr = DMDestroy(&odm);CHKERRQ(ierr);
    ierr = DMDestroy(&rdm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm;
  AppCtx         user;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = ProcessOptions(PETSC_COMM_WORLD, &user);CHKERRQ(ierr);
  ierr = CreateMesh(PETSC_COMM_WORLD, &user, &dm);CHKERRQ(ierr);
  if (user.initOrnt) {
    ierr = ReorientCell(dm, 0, user.initOrnt, PETSC_FALSE);CHKERRQ(ierr);
    ierr = DMViewFromOptions(dm, NULL, "-ornt_dm_view");CHKERRQ(ierr);
  }
  ierr = GenerateArrangments(dm, &user);CHKERRQ(ierr);
  ierr = VerifyCayleyTable(dm, &user);CHKERRQ(ierr);
  ierr = VerifyInverse(dm, &user);CHKERRQ(ierr);
  ierr = RefineArrangments(dm, &user);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: segment
    args: -dm_plex_reference_cell_domain -dm_plex_cell segment -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0 -dm_plex_view_colors_depth 1,0 -dm_plex_view_tikzscale 0.5

  test:
    suffix: triangle
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 0.5

  test:
    suffix: quadrilateral
    args: -dm_plex_reference_cell_domain -dm_plex_cell quadrilateral -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 0.5

  test:
    suffix: tensor_segment
    args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_quad -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 0.5

  test:
    suffix: tetrahedron
    args: -dm_plex_reference_cell_domain -dm_plex_cell tetrahedron -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 0.5

  test:
    suffix: hexahedron
    args: -dm_plex_reference_cell_domain -dm_plex_cell hexahedron -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 0.3

  test:
    suffix: triangular_prism
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangular_prism -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 0.5

  test:
    suffix: tensor_triangular_prism
    args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_triangular_prism -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 0.5

  test:
    suffix: tensor_quadrilateral_prism
    args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_quadrilateral_prism -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 0.5

  test:
    suffix: pyramid
    args: -dm_plex_reference_cell_domain -dm_plex_cell pyramid -gen_arrangements \
          -gen_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 0.5

  test:
    suffix: ref_segment
    args: -dm_plex_reference_cell_domain -dm_plex_cell segment -ref_arrangements -dm_plex_check_all \
          -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0 -dm_plex_view_colors_depth 1,0 -dm_plex_view_tikzscale 1.0

  test:
    suffix: ref_triangle
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangle -ref_arrangements -dm_plex_check_all \
          -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

  test:
    suffix: ref_quadrilateral
    args: -dm_plex_reference_cell_domain -dm_plex_cell quadrilateral -ref_arrangements -dm_plex_check_all \
          -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

  test:
    suffix: ref_tensor_segment
    args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_quad -ref_arrangements -dm_plex_check_all \
          -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,1,0 -dm_plex_view_colors_depth 1,1,0 -dm_plex_view_tikzscale 1.0

  test:
    suffix: ref_tetrahedron
    args: -dm_plex_reference_cell_domain -dm_plex_cell tetrahedron -ref_arrangements -dm_plex_check_all \
          -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

  test:
    suffix: ref_hexahedron
    args: -dm_plex_reference_cell_domain -dm_plex_cell hexahedron -ref_arrangements -dm_plex_check_all \
          -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

  test:
    suffix: ref_triangular_prism
    args: -dm_plex_reference_cell_domain -dm_plex_cell triangular_prism -ref_arrangements -dm_plex_check_all \
          -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

  test:
    suffix: ref_tensor_triangular_prism
    args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_triangular_prism -ref_arrangements -dm_plex_check_all \
          -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

  test:
    suffix: ref_tensor_quadrilateral_prism
    args: -dm_plex_reference_cell_domain -dm_plex_cell tensor_quadrilateral_prism -ref_arrangements -dm_plex_check_all \
          -ref_dm_view ::ascii_latex -dm_plex_view_numbers_depth 1,0,0,0 -dm_plex_view_colors_depth 1,0,0,0 -dm_plex_view_tikzscale 1.0

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
