static char help[] = "Tests dof numberings for external integrators such as LibCEED.\n\n";

#include <petscdmplex.h>
#include <petscds.h>

typedef struct {
  PetscBool useFE;
  PetscInt  check_face;
  PetscBool closure_tensor;
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscFunctionBeginUser;
  options->useFE          = PETSC_TRUE;
  options->check_face     = 1;
  options->closure_tensor = PETSC_FALSE;
  PetscOptionsBegin(comm, "", "Dof Ordering Options", "DMPLEX");
  PetscCall(PetscOptionsBool("-use_fe", "Use FE or FV discretization", "ex49.c", options->useFE, &options->useFE, NULL));
  PetscCall(PetscOptionsInt("-check_face", "Face set to report on", "ex49.c", options->check_face, &options->check_face, NULL));
  PetscCall(PetscOptionsBool("-closure_tensor", "Use DMPlexSetClosurePermutationTensor()", "ex49.c", options->closure_tensor, &options->closure_tensor, NULL));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, AppCtx *user, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMSetApplicationContext(*dm, user));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode SetupDiscretization(DM dm, AppCtx *user)
{
  DM       cdm = dm;
  PetscInt dim;

  PetscFunctionBeginUser;
  PetscCall(DMGetDimension(dm, &dim));
  if (user->useFE) {
    PetscFE fe;

    PetscCall(PetscFECreateDefault(PETSC_COMM_SELF, dim, 1, PETSC_FALSE, NULL, -1, &fe));
    PetscCall(PetscObjectSetName((PetscObject)fe, "scalar"));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fe));
    PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fe));
    PetscCall(PetscFEDestroy(&fe));
  } else {
    PetscFV fv;

    PetscCall(PetscFVCreate(PETSC_COMM_SELF, &fv));
    PetscCall(PetscFVSetType(fv, PETSCFVLEASTSQUARES));
    PetscCall(PetscFVSetNumComponents(fv, dim));
    PetscCall(PetscFVSetSpatialDimension(fv, dim));
    PetscCall(PetscFVSetFromOptions(fv));
    PetscCall(PetscFVSetUp(fv));
    PetscCall(PetscObjectSetName((PetscObject)fv, "vector"));
    PetscCall(DMSetField(dm, 0, NULL, (PetscObject)fv));
    PetscCall(DMSetField(dm, 1, NULL, (PetscObject)fv));
    PetscCall(PetscFVDestroy(&fv));
  }
  PetscCall(DMCreateDS(dm));
  while (cdm) {
    PetscCall(DMCopyDisc(dm, cdm));
    PetscCall(DMGetCoarseDM(cdm, &cdm));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CheckOffsets(DM dm, AppCtx *user, const char *domain_name, PetscInt label_value, PetscInt height)
{
  const char            *height_name[] = {"cells", "faces"};
  DMLabel                domain_label  = NULL;
  DM                     cdm;
  PetscInt               Nf, f;
  ISLocalToGlobalMapping ltog;

  PetscFunctionBeginUser;
  if (domain_name) PetscCall(DMGetLabel(dm, domain_name, &domain_label));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "## %s: '%s' {%" PetscInt_FMT "}%s\n", height_name[height], domain_name ? domain_name : "default", label_value, domain_name && !domain_label ? " (null label)" : ""));
  if (domain_name && !domain_label) PetscFunctionReturn(PETSC_SUCCESS);
  if (user->closure_tensor) PetscCall(DMPlexSetClosurePermutationTensor(dm, PETSC_DETERMINE, NULL));
  // Offsets for cell closures
  PetscCall(DMGetNumFields(dm, &Nf));
  for (f = 0; f < Nf; ++f) {
    PetscObject  obj;
    PetscClassId id;
    char         name[PETSC_MAX_PATH_LEN];

    PetscCall(DMGetField(dm, f, NULL, &obj));
    PetscCall(PetscObjectGetClassId(obj, &id));
    if (id == PETSCFE_CLASSID) {
      IS        offIS;
      PetscInt *offsets, Ncell, Ncl, Nc, n;

      PetscCall(DMPlexGetLocalOffsets(dm, domain_label, label_value, height, f, &Ncell, &Ncl, &Nc, &n, &offsets));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, Ncell * Ncl, offsets, PETSC_OWN_POINTER, &offIS));
      PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "Field %" PetscInt_FMT " Offsets", f));
      PetscCall(PetscObjectSetName((PetscObject)offIS, name));
      PetscCall(ISViewFromOptions(offIS, NULL, "-offsets_view"));
      PetscCall(ISDestroy(&offIS));
    } else if (id == PETSCFV_CLASSID) {
      IS        offIS;
      PetscInt *offsets, *offsetsNeg, *offsetsPos, Nface, Nc, n;

      PetscCall(DMPlexGetLocalOffsetsSupport(dm, domain_label, label_value, &Nface, &Nc, &n, &offsetsNeg, &offsetsPos));
      PetscCall(PetscMalloc1(Nface * Nc * 2, &offsets));
      for (PetscInt f = 0, i = 0; f < Nface; ++f) {
        for (PetscInt c = 0; c < Nc; ++c) offsets[i++] = offsetsNeg[f * Nc + c];
        for (PetscInt c = 0; c < Nc; ++c) offsets[i++] = offsetsPos[f * Nc + c];
      }
      PetscCall(PetscFree(offsetsNeg));
      PetscCall(PetscFree(offsetsPos));
      PetscCall(ISCreateGeneral(PETSC_COMM_SELF, Nface * Nc * 2, offsets, PETSC_OWN_POINTER, &offIS));
      PetscCall(PetscSNPrintf(name, PETSC_MAX_PATH_LEN, "Field %" PetscInt_FMT " Offsets", f));
      PetscCall(PetscObjectSetName((PetscObject)offIS, name));
      PetscCall(ISViewFromOptions(offIS, NULL, "-offsets_view"));
      PetscCall(ISDestroy(&offIS));
    } else SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Unrecognized type for DM field %" PetscInt_FMT, f);
  }
  PetscCall(DMGetLocalToGlobalMapping(dm, &ltog));
  PetscCall(ISLocalToGlobalMappingViewFromOptions(ltog, NULL, "-ltog_view"));

  // Offsets for coordinates
  {
    Vec                X;
    PetscSection       s;
    const PetscScalar *x;
    const char        *cname;
    PetscInt           cdim, *offsets, Ncell, Ncl, Nc, n;
    PetscBool          isDG = PETSC_FALSE;

    PetscCall(DMGetCellCoordinateDM(dm, &cdm));
    if (!cdm) {
      PetscCall(DMGetCoordinateDM(dm, &cdm));
      cname = "Coordinates";
      PetscCall(DMGetCoordinatesLocal(dm, &X));
    } else {
      isDG  = PETSC_TRUE;
      cname = "DG Coordinates";
      PetscCall(DMGetCellCoordinatesLocal(dm, &X));
    }
    if (isDG && height) PetscFunctionReturn(PETSC_SUCCESS);
    if (domain_name) PetscCall(DMGetLabel(cdm, domain_name, &domain_label));
    if (user->closure_tensor) PetscCall(DMPlexSetClosurePermutationTensor(cdm, PETSC_DETERMINE, NULL));
    PetscCall(DMPlexGetLocalOffsets(cdm, domain_label, label_value, height, 0, &Ncell, &Ncl, &Nc, &n, &offsets));
    PetscCall(DMGetCoordinateDim(dm, &cdim));
    PetscCheck(Nc == cdim, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Geometric dimension %" PetscInt_FMT " should be %" PetscInt_FMT, Nc, cdim);
    PetscCall(DMGetLocalSection(cdm, &s));
    PetscCall(VecGetArrayRead(X, &x));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%s by element in %s order\n", cname, user->closure_tensor ? "tensor" : "bfs"));
    for (PetscInt c = 0; c < Ncell; ++c) {
      for (PetscInt v = 0; v < Ncl; ++v) {
        PetscInt           off = offsets[c * Ncl + v], dgdof;
        const PetscScalar *vx  = &x[off];

        if (isDG) {
          PetscCall(PetscSectionGetDof(s, c, &dgdof));
          PetscCheck(Ncl * Nc == dgdof, PETSC_COMM_SELF, PETSC_ERR_ARG_INCOMP, "Offset size %" PetscInt_FMT " should be %" PetscInt_FMT, Ncl * Nc, dgdof);
        }
        switch (cdim) {
        case 1:
          PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%" PetscInt_FMT "] %" PetscInt_FMT " <-- %2" PetscInt_FMT " (% 4.2f)\n", c, v, off, (double)PetscRealPart(vx[0])));
          break;
        case 2:
          PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%" PetscInt_FMT "] %" PetscInt_FMT " <-- %2" PetscInt_FMT " (% 4.2f, % 4.2f)\n", c, v, off, (double)PetscRealPart(vx[0]), (double)PetscRealPart(vx[1])));
          break;
        case 3:
          PetscCall(PetscSynchronizedPrintf(PETSC_COMM_WORLD, "[%" PetscInt_FMT "] %" PetscInt_FMT " <-- %2" PetscInt_FMT " (% 4.2f, % 4.2f, % 4.2f)\n", c, v, off, (double)PetscRealPart(vx[0]), (double)PetscRealPart(vx[1]), (double)PetscRealPart(vx[2])));
        }
      }
    }
    PetscCall(PetscSynchronizedFlush(PETSC_COMM_WORLD, stdout));
    PetscCall(VecRestoreArrayRead(X, &x));
    PetscCall(PetscFree(offsets));
    PetscCall(DMGetLocalToGlobalMapping(cdm, &ltog));
    PetscCall(ISLocalToGlobalMappingViewFromOptions(ltog, NULL, "-coord_ltog_view"));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM       dm;
  AppCtx   user;
  PetscInt depth;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(ProcessOptions(PETSC_COMM_WORLD, &user));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &user, &dm));
  PetscCall(SetupDiscretization(dm, &user));
  PetscCall(CheckOffsets(dm, &user, NULL, 0, 0));
  PetscCall(DMPlexGetDepth(dm, &depth));
  if (depth > 1) PetscCall(CheckOffsets(dm, &user, "Face Sets", user.check_face, 1));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -dm_refine 1 -petscspace_degree 1 -dm_view -offsets_view

  test:
    suffix: 1
    args: -dm_plex_simplex 0 -dm_plex_box_bd periodic,none -dm_plex_box_faces 3,3 -dm_sparse_localize 0 -petscspace_degree 1 \
          -dm_view -offsets_view

  test:
    suffix: cg_2d
    args: -dm_plex_simplex 0 -dm_plex_box_bd none,none -dm_plex_box_faces 3,3 -petscspace_degree 1 \
          -dm_view -offsets_view

  test:
    suffix: 1d_sfc
    args: -dm_plex_simplex 0 -dm_plex_dim 1 -dm_plex_shape zbox -dm_plex_box_faces 3 1 -dm_view -coord_ltog_view

  test:
    suffix: 2d_sfc
    nsize: 2
    args: -dm_plex_simplex 0 -dm_plex_dim 2 -dm_plex_shape zbox -dm_plex_box_faces 4,3 -dm_distribute 0 -petscspace_degree 1 -dm_view

  test:
    suffix: 2d_sfc_periodic
    nsize: 2
    args: -dm_plex_simplex 0 -dm_plex_dim 2 -dm_plex_shape zbox -dm_plex_box_faces 4,3 -dm_distribute 0 -petscspace_degree 1 -dm_plex_box_bd periodic,none -dm_view ::ascii_info_detail

  testset:
    args: -dm_plex_simplex 0 -dm_plex_dim 2 -dm_plex_shape zbox -dm_plex_box_faces 3,2 -petscspace_degree 1 -dm_plex_box_bd none,periodic -dm_view ::ascii_info_detail -closure_tensor
    nsize: 2
    test:
      suffix: 2d_sfc_periodic_stranded
      args: -dm_distribute 0
    test:
      suffix: 2d_sfc_periodic_stranded_dist
      args: -dm_distribute 1 -petscpartitioner_type simple

  test:
    suffix: fv_0
    requires: triangle
    args: -dm_refine 1 -use_fe 0 -dm_view -offsets_view

TEST*/
