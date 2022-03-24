static char help[] = "Tests for creation of submeshes\n\n";

#include <petscdmplex.h>

PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  CHKERRQ(DMCreate(comm, dm));
  CHKERRQ(DMSetType(*dm, DMPLEX));
  CHKERRQ(DMSetFromOptions(*dm));
  CHKERRQ(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSubmesh(DM dm, PetscBool start, DM *subdm)
{
  DMLabel        label, map;
  PetscInt       cStart, cEnd, cStartSub, cEndSub, c;

  PetscFunctionBegin;
  CHKERRQ(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  CHKERRQ(DMCreateLabel(dm, "cells"));
  CHKERRQ(DMGetLabel(dm, "cells", &label));
  CHKERRQ(DMLabelClearStratum(label, 1));
  if (start) {cStartSub = cStart; cEndSub = cEnd/2;}
  else       {cStartSub = cEnd/2; cEndSub = cEnd;}
  for (c = cStartSub; c < cEndSub; ++c) CHKERRQ(DMLabelSetValue(label, c, 1));
  CHKERRQ(DMPlexFilter(dm, label, 1, subdm));
  CHKERRQ(PetscObjectSetName((PetscObject) *subdm, "Submesh"));
  CHKERRQ(DMViewFromOptions(*subdm, NULL, "-dm_view"));
  CHKERRQ(DMPlexGetSubpointMap(*subdm, &map));
  CHKERRQ(DMLabelView(map, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, subdm;

  CHKERRQ(PetscInitialize(&argc, &argv, NULL, help));
  CHKERRQ(CreateMesh(PETSC_COMM_WORLD, &dm));
  CHKERRQ(CreateSubmesh(dm, PETSC_TRUE, &subdm));
  CHKERRQ(DMSetFromOptions(subdm));
  CHKERRQ(DMDestroy(&subdm));
  CHKERRQ(CreateSubmesh(dm, PETSC_FALSE, &subdm));
  CHKERRQ(DMSetFromOptions(subdm));
  CHKERRQ(DMDestroy(&subdm));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -dm_coord_space 0 -dm_view ascii::ascii_info_detail -dm_plex_check_all

TEST*/
