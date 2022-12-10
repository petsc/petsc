static char help[] = "Tests for ephemeral meshes.\n";

#include <petscdmplex.h>
#include <petscdmplextransform.h>
#include <petscdmlabelephemeral.h>

/*
  Use

    -ref_dm_view -eph_dm_view

  to view the concrete and ephemeral meshes from the first transformation, and

   -ref_dm_sec_view -eph_dm_sec_view

  for the second.
*/

// Should remove when I have an API for everything
#include <petsc/private/dmplextransformimpl.h>

typedef struct {
  DMLabel   active;   /* Label for transform */
  PetscBool second;   /* Flag to execute a second transformation */
  PetscBool concrete; /* Flag to use the concrete mesh for the second transformation */
} AppCtx;

static PetscErrorCode ProcessOptions(MPI_Comm comm, AppCtx *options)
{
  PetscInt  cells[1024], Nc = 1024;
  PetscBool flg;

  PetscFunctionBeginUser;
  options->active   = NULL;
  options->second   = PETSC_FALSE;
  options->concrete = PETSC_FALSE;

  PetscOptionsBegin(comm, "", "Ephemeral Meshing Options", "DMPLEX");
  PetscCall(PetscOptionsIntArray("-cells", "Cells to mark for transformation", "ex57.c", cells, &Nc, &flg));
  if (flg) {
    PetscCall(DMLabelCreate(comm, "active", &options->active));
    for (PetscInt c = 0; c < Nc; ++c) PetscCall(DMLabelSetValue(options->active, cells[c], DM_ADAPT_REFINE));
  }
  PetscCall(PetscOptionsBool("-second", "Use a second transformation", "ex57.c", options->second, &options->second, &flg));
  PetscCall(PetscOptionsBool("-concrete", "Use concrete mesh for the second transformation", "ex57.c", options->concrete, &options->concrete, &flg));
  PetscOptionsEnd();
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(PetscObjectSetName((PetscObject)*dm, "Mesh"));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateTransform(DM dm, DMLabel active, const char prefix[], DMPlexTransform *tr)
{
  PetscFunctionBeginUser;
  PetscCall(DMPlexTransformCreate(PetscObjectComm((PetscObject)dm), tr));
  PetscCall(PetscObjectSetName((PetscObject)*tr, "Transform"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*tr, prefix));
  PetscCall(DMPlexTransformSetDM(*tr, dm));
  PetscCall(DMPlexTransformSetActive(*tr, active));
  PetscCall(DMPlexTransformSetFromOptions(*tr));
  PetscCall(DMPlexTransformSetUp(*tr));

  PetscCall(DMSetApplicationContext(dm, *tr));
  PetscCall(PetscObjectViewFromOptions((PetscObject)*tr, NULL, "-dm_plex_transform_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateEphemeralMesh(DMPlexTransform tr, DM *tdm)
{
  PetscFunctionBegin;
  PetscCall(DMPlexCreateEphemeral(tr, tdm));
  PetscCall(PetscObjectSetName((PetscObject)*tdm, "Ephemeral Mesh"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*tdm, "eph_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateConcreteMesh(DMPlexTransform tr, DM *rdm)
{
  DM cdm, codm, rcodm;

  PetscFunctionBegin;
  PetscCall(DMPlexTransformGetDM(tr, &cdm));
  PetscCall(DMPlexTransformApply(tr, cdm, rdm));
  PetscCall(DMSetCoarsenLevel(*rdm, cdm->leveldown));
  PetscCall(DMSetRefineLevel(*rdm, cdm->levelup + 1));
  PetscCall(DMCopyDisc(cdm, *rdm));
  PetscCall(DMGetCoordinateDM(cdm, &codm));
  PetscCall(DMGetCoordinateDM(*rdm, &rcodm));
  PetscCall(DMCopyDisc(codm, rcodm));
  PetscCall(DMPlexTransformCreateDiscLabels(tr, *rdm));
  PetscCall(DMSetCoarseDM(*rdm, cdm));
  PetscCall(DMPlexSetRegularRefinement(*rdm, PETSC_TRUE));
  if (rdm) {
    ((DM_Plex *)(*rdm)->data)->printFEM = ((DM_Plex *)cdm->data)->printFEM;
    ((DM_Plex *)(*rdm)->data)->printL2  = ((DM_Plex *)cdm->data)->printL2;
  }
  PetscCall(PetscObjectSetName((PetscObject)*rdm, "Concrete Mesh"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*rdm, "ref_"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CompareMeshes(DM dmA, DM dmB, DM dm)
{
  PetscInt dim, dimB, pStart, pEnd, pStartB, pEndB;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscCall(PetscObjectGetComm((PetscObject)dmA, &comm));
  PetscCall(DMGetDimension(dmA, &dim));
  PetscCall(DMGetDimension(dmB, &dimB));
  PetscCheck(dim == dimB, comm, PETSC_ERR_ARG_INCOMP, "Dimension from dmA %" PetscInt_FMT " != %" PetscInt_FMT " from dmB", dim, dimB);
  PetscCall(DMPlexGetChart(dmA, &pStart, &pEnd));
  PetscCall(DMPlexGetChart(dmB, &pStartB, &pEndB));
  PetscCheck(pStart == pStartB && pEnd == pEndB, comm, PETSC_ERR_ARG_INCOMP, "Chart from dmA (%" PetscInt_FMT ", %" PetscInt_FMT ") does not match (%" PetscInt_FMT ", %" PetscInt_FMT ") for dmB", pStart, pEnd, pStartB, pEndB);
  for (PetscInt p = pStart; p < pEnd; ++p) {
    const PetscInt *cone, *ornt, *coneB, *orntB;
    PetscInt        coneSize, coneSizeB;

    PetscCall(DMPlexGetConeSize(dmA, p, &coneSize));
    PetscCall(DMPlexGetConeSize(dmB, p, &coneSizeB));
    PetscCheck(coneSize == coneSizeB, comm, PETSC_ERR_ARG_INCOMP, "Cone size for %" PetscInt_FMT " from dmA %" PetscInt_FMT " does not match %" PetscInt_FMT " for dmB", p, coneSize, coneSizeB);
    PetscCall(DMPlexGetOrientedCone(dmA, p, &cone, &ornt));
    PetscCall(DMPlexGetOrientedCone(dmB, p, &coneB, &orntB));
    for (PetscInt c = 0; c < coneSize; ++c) {
      PetscCheck(cone[c] == coneB[c] && ornt[c] == orntB[c], comm, PETSC_ERR_ARG_INCOMP, "Cone point %" PetscInt_FMT " for point %" PetscInt_FMT " from dmA (%" PetscInt_FMT ", %" PetscInt_FMT ") does not match (%" PetscInt_FMT ", %" PetscInt_FMT ") for dmB", c, p, cone[c], ornt[c], coneB[c], orntB[c]);
    }
    PetscCall(DMPlexRestoreOrientedCone(dmA, p, &cone, &ornt));
    PetscCall(DMPlexRestoreOrientedCone(dmB, p, &coneB, &orntB));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char *argv[])
{
  DM              dm, tdm, rdm;
  DMPlexTransform tr;
  AppCtx          user;
  MPI_Comm        comm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  comm = PETSC_COMM_WORLD;
  PetscCall(ProcessOptions(comm, &user));
  PetscCall(CreateMesh(comm, &dm));
  PetscCall(CreateTransform(dm, user.active, "first_", &tr));
  PetscCall(CreateEphemeralMesh(tr, &tdm));
  PetscCall(CreateConcreteMesh(tr, &rdm));
  if (user.second) {
    DMPlexTransform tr2;
    DM              tdm2, rdm2;

    PetscCall(DMViewFromOptions(rdm, NULL, "-dm_sec_view"));
    PetscCall(DMViewFromOptions(tdm, NULL, "-dm_sec_view"));
    if (user.concrete) PetscCall(CreateTransform(rdm, user.active, "second_", &tr2));
    else PetscCall(CreateTransform(tdm, user.active, "second_", &tr2));
    PetscCall(CreateEphemeralMesh(tr2, &tdm2));
    PetscCall(CreateConcreteMesh(tr2, &rdm2));
    PetscCall(DMDestroy(&tdm));
    PetscCall(DMDestroy(&rdm));
    PetscCall(DMPlexTransformDestroy(&tr2));
    tdm = tdm2;
    rdm = rdm2;
  }
  PetscCall(DMViewFromOptions(tdm, NULL, "-dm_view"));
  PetscCall(DMViewFromOptions(rdm, NULL, "-dm_view"));
  PetscCall(CompareMeshes(rdm, tdm, dm));
  PetscCall(DMPlexTransformDestroy(&tr));
  PetscCall(DMDestroy(&tdm));
  PetscCall(DMDestroy(&rdm));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMLabelDestroy(&user.active));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  # Tests for regular refinement of whole meshes
  test:
    suffix: tri
    requires: triangle
    args: -first_dm_plex_transform_view ::ascii_info_detail

  test:
    suffix: quad
    args: -dm_plex_simplex 0

  # Here I am checking that the 'marker' label is correct for the ephemeral mesh
  test:
    suffix: tet
    requires: ctetgen
    args: -dm_plex_dim 3 -ref_dm_view -eph_dm_view -eph_dm_plex_view_labels marker

  test:
    suffix: hex
    args: -dm_plex_dim 3 -dm_plex_simplex 0

  # Tests for filter patches
  testset:
    args: -first_dm_plex_transform_type transform_filter -ref_dm_view

    test:
      suffix: tri_patch
      requires: triangle
      args: -cells 0,1,2,4

    test:
      suffix: quad_patch
      args: -dm_plex_simplex 0 -dm_plex_box_faces 3,3 -cells 0,1,3,4

  # Tests for refined filter patches
  testset:
    args: -first_dm_plex_transform_type transform_filter -ref_dm_view -eph_dm_view -second

    test:
      suffix: tri_patch_ref
      requires: triangle
      args: -cells 0,1,2,4

    test:
      suffix: tri_patch_ref_concrete
      requires: triangle
      args: -cells 0,1,2,4 -concrete -first_dm_plex_transform_view ::ascii_info_detail

  # Tests for boundary layer refinement
  test:
    suffix: quad_bl
    args: -dm_plex_simplex 0 -dm_plex_dim 1 -dm_plex_box_faces 5 -dm_extrude 2 -cells 0,2,4,6,8 \
          -first_dm_plex_transform_type refine_boundary_layer -first_dm_plex_transform_bl_splits 4 \
          -ref_dm_view

TEST*/
