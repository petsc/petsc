static char help[] = "Tests for creation of submeshes\n\n";

#include <petscdmplex.h>

static PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBeginUser;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
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
  if (lower) {cStartSub = cStart; cEndSub = cEnd/2;}
  else       {cStartSub = cEnd/2; cEndSub = cEnd;}
  for (PetscInt c = cStartSub; c < cEndSub; ++c) PetscCall(DMLabelSetValue(*label, c, 1));
  PetscCall(DMPlexLabelComplete(dm, *label));
  PetscFunctionReturn(0);
}

// Label everything on the right half of the domain
static PetscErrorCode CreateHalfDomainLabel(DM dm, PetscBool lower, DMLabel *label)
{
  PetscReal centroid[3];
  PetscInt  cStart, cEnd, cdim;

  PetscFunctionBeginUser;
  PetscCall(DMCreateLabel(dm, "cells"));
  PetscCall(DMGetLabel(dm, "cells", label));
  PetscCall(DMLabelClearStratum(*label, 1));
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMGetCoordinateDim(dm, &cdim));
  for (PetscInt c = cStart; c < cEnd; ++c) {
    PetscCall(DMPlexComputeCellGeometryFVM(dm, c, NULL, centroid, NULL));
    if (lower) {if (centroid[0] < 0.5) PetscCall(DMLabelSetValue(*label, c, 1));}
    else       {if (centroid[0] > 0.5) PetscCall(DMLabelSetValue(*label, c, 1));}
  }
  PetscCall(DMPlexLabelComplete(dm, *label));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSubmesh(DM dm, PetscBool domain, PetscBool lower, DM *subdm)
{
  DMLabel label, map;

  PetscFunctionBegin;
  if (domain) PetscCall(CreateHalfDomainLabel(dm, lower, &label));
  else        PetscCall(CreateHalfCellsLabel(dm, lower, &label));
  PetscCall(DMPlexFilter(dm, label, 1, subdm));
  PetscCall(PetscObjectSetName((PetscObject) *subdm, "Submesh"));
  PetscCall(PetscObjectSetOptionsPrefix((PetscObject) *subdm, "sub_"));
  PetscCall(DMViewFromOptions(*subdm, NULL, "-dm_view"));
  PetscCall(DMPlexGetSubpointMap(*subdm, &map));
  PetscCall(PetscObjectViewFromOptions((PetscObject) map, NULL, "-map_view"));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM        dm, subdm;
  PetscBool domain = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-domain", &domain, NULL));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(CreateSubmesh(dm, domain, PETSC_TRUE, &subdm));
  PetscCall(DMSetFromOptions(subdm));
  PetscCall(DMDestroy(&subdm));
  PetscCall(CreateSubmesh(dm, domain, PETSC_FALSE, &subdm));
  PetscCall(DMSetFromOptions(subdm));
  PetscCall(DMDestroy(&subdm));
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

TEST*/
