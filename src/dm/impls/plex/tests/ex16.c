static char help[] = "Tests for creation of submeshes\n\n";

#include <petscdmplex.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Label half of the cells
static PetscErrorCode CreateHalfCellsLabel(DM dm, PetscBool lower, DMLabel *label)
{
  PetscInt cStart, cEnd, cStartSub, cEndSub;

  PetscFunctionBeginUser;
  PetscCall(DMCreateLabel(dm, "cells"));
  PetscCall(DMGetLabel(dm, "cells", label));
  PetscCall(DMLabelClearStratum(*label, 1));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  if (lower) {
    cStartSub = cStart;
    cEndSub   = cEnd / 2;
  } else {
    cStartSub = cEnd / 2;
    cEndSub   = cEnd;
  }
  for (PetscInt c = cStartSub; c < cEndSub; ++c) PetscCall(DMLabelSetValue(*label, c, 1));
  PetscCall(DMPlexLabelComplete(dm, *label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Label everything on the right half of the domain
static PetscErrorCode CreateHalfDomainLabel(DM dm, PetscBool lower, PetscReal height, DMLabel *label)
{
  PetscReal centroid[3];
  PetscInt  cStart, cEnd, cdim;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMCreateLabel(dm, "cells"));
  PetscCall(DMGetLabel(dm, "cells", label));
  PetscCall(DMLabelClearStratum(*label, 1));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
    if (height > 0.0 && PetscAbsReal(centroid[1] - height) > PETSC_SMALL) continue;
    if (lower) {
      if (centroid[0] < 0.5) PetscCall(DMLabelSetValue(*label, c, 1));
    } else {
      if (centroid[0] > 0.5) PetscCall(DMLabelSetValue(*label, c, 1));
    }
  }
  PetscCall(DMPlexLabelComplete(dm, *label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

// Create a line of faces at a given x value
static PetscErrorCode CreateLineLabel(DM dm, PetscReal x, DMLabel *label)
{
  PetscReal centroid[3];
  PetscInt  fStart, fEnd;

  PetscFunctionBeginUser;
  PetscCall(DMGetCoordinatesLocalSetUp(dm));
  PetscCall(DMCreateLabel(dm, "faces"));
  PetscCall(DMGetLabel(dm, "faces", label));
  PetscCall(DMPlexGetHeightStratum(dm, 1, &fStart, &fEnd));
  for (PetscInt f = fStart; f < fEnd; ++f) {
    PetscCall(DMPlexComputeCellGeometryFVM(dm, f, NULL, centroid, NULL));
    if (PetscAbsReal(centroid[0] - x) < PETSC_SMALL) PetscCall(DMLabelSetValue(*label, f, 1));
  }
  PetscCall(DMPlexLabelComplete(dm, *label));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode CreateVolumeSubmesh(DM dm, PetscBool domain, PetscBool lower, PetscReal height, DM *subdm)
{
  DMLabel label, map;

  PetscFunctionBegin;
  if (domain) PetscCall(CreateHalfDomainLabel(dm, lower, height, &label));
  else PetscCall(CreateHalfCellsLabel(dm, lower, &label));
  PetscCall(DMPlexFilter(dm, label, 1, subdm));
  PetscCall(PetscObjectSetName((PetscObject)*subdm, "Submesh"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)*subdm, "sub_"));
  PetscCall(DMViewFromOptions(*subdm, NULL, "-dm_view"));
  PetscCall(DMPlexGetSubpointMap(*subdm, &map));
  PetscCall(PetscObjectViewFromOptions((PetscObject)map, NULL, "-map_view"));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode TestBoundaryField(DM dm)
{
  DM       subdm;
  DMLabel  label, map;
  PetscFV  fvm;
  Vec      gv;
  PetscInt cdim;

  PetscFunctionBeginUser;
  PetscCall(CreateLineLabel(dm, 0.5, &label));
  PetscCall(DMPlexFilter(dm, label, 1, &subdm));
  PetscCall(PetscObjectSetName((PetscObject)subdm, "Submesh"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject)subdm, "sub_"));
  PetscCall(DMViewFromOptions(subdm, NULL, "-dm_view"));
  PetscCall(DMPlexGetSubpointMap(subdm, &map));
  PetscCall(PetscObjectViewFromOptions((PetscObject)map, NULL, "-map_view"));

  PetscCall(PetscFVCreate(PetscObjectComm((PetscObject)subdm), &fvm));
  PetscCall(PetscObjectSetName((PetscObject)fvm, "testField"));
  PetscCall(PetscFVSetNumComponents(fvm, 1));
  PetscCall(DMGetCoordinateDim(subdm, &cdim));
  PetscCall(PetscFVSetSpatialDimension(fvm, cdim));
  PetscCall(PetscFVSetFromOptions(fvm));

  PetscCall(DMAddField(subdm, NULL, (PetscObject)fvm));
  PetscCall(PetscFVDestroy(&fvm));
  PetscCall(DMCreateDS(subdm));

  PetscCall(DMCreateGlobalVector(subdm, &gv));
  PetscCall(PetscObjectSetName((PetscObject)gv, "potential"));
  PetscCall(VecSet(gv, 1.));
  PetscCall(VecViewFromOptions(gv, NULL, "-vec_view"));
  PetscCall(VecDestroy(&gv));
  PetscCall(DMDestroy(&subdm));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  DM        dm, subdm;
  PetscReal height = -1.0;
  PetscBool volume = PETSC_TRUE, domain = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-height", &height, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-volume", &volume, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-domain", &domain, NULL));

  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  if (volume) {
    PetscCall(CreateVolumeSubmesh(dm, domain, PETSC_TRUE, height, &subdm));
    PetscCall(DMSetFromOptions(subdm));
    PetscCall(DMDestroy(&subdm));
    PetscCall(CreateVolumeSubmesh(dm, domain, PETSC_FALSE, height, &subdm));
    PetscCall(DMSetFromOptions(subdm));
    PetscCall(DMDestroy(&subdm));
  } else {
    PetscCall(TestBoundaryField(dm));
  }
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -dm_coord_space 0 -sub_dm_plex_check_all \
          -dm_view ascii::ascii_info_detail -sub_dm_view ascii::ascii_info_detail -map_view

  # These tests check that filtering is stable when boundary point ownership could change, so it needs 3 processes
  testset:
    nsize: 3
    requires: parmetis
    args: -dm_plex_simplex 0 -dm_plex_box_faces 20,20 -petscpartitioner_type parmetis -dm_distribute_overlap 1 -sub_dm_distribute 0 \
          -sub_dm_plex_check_all -dm_view -sub_dm_view

    test:
      suffix: 1

    test:
      suffix: 2
      args: -domain

  # This set tests that global numberings can be made when some strata are missing on a process
  testset:
    nsize: 3
    requires: hdf5
    args: -dm_plex_simplex 0 -dm_plex_box_faces 4,4 -petscpartitioner_type simple -sub_dm_distribute 0 \
          -sub_dm_plex_check_all -sub_dm_view hdf5:subdm.h5

    test:
      suffix: 3
      args: -domain -height 0.625

  # This test checks whether filter can extract a lower-dimensional manifold and output a field on it
  testset:
    args: -volume 0 -dm_plex_simplex 0 -sub_dm_view hdf5:subdm.h5 -vec_view hdf5:subdm.h5::append
    requires: hdf5

    test:
      suffix: surface_2d
      args: -dm_plex_box_faces 5,5

    test:
      suffix: surface_3d
      args: -dm_plex_box_faces 5,5,5

    # This test checks that if cells are present in the SF, the dm is marked as having an overlap
    test:
      suffix: surface_2d_2
      nsize: 3
      args: -dm_plex_box_faces 5,5 -domain -height 0.625

TEST*/
