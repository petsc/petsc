static char help[] = "Tests for creation of submeshes\n\n";

#include <petscdmplex.h>

PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscFunctionBegin;
  PetscCall(DMCreate(comm, dm));
  PetscCall(DMSetType(*dm, DMPLEX));
  PetscCall(DMSetFromOptions(*dm));
  PetscCall(DMViewFromOptions(*dm, NULL, "-dm_view"));
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSubmesh(DM dm, PetscBool start, DM *subdm)
{
  DMLabel        label, map;
  PetscInt       cStart, cEnd, cStartSub, cEndSub, c;

  PetscFunctionBegin;
  PetscCall(DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd));
  PetscCall(DMCreateLabel(dm, "cells"));
  PetscCall(DMGetLabel(dm, "cells", &label));
  PetscCall(DMLabelClearStratum(label, 1));
  if (start) {cStartSub = cStart; cEndSub = cEnd/2;}
  else       {cStartSub = cEnd/2; cEndSub = cEnd;}
  for (c = cStartSub; c < cEndSub; ++c) PetscCall(DMLabelSetValue(label, c, 1));
  PetscCall(DMPlexFilter(dm, label, 1, subdm));
  PetscCall(PetscObjectSetName((PetscObject) *subdm, "Submesh"));
  PetscCall(DMViewFromOptions(*subdm, NULL, "-dm_view"));
  PetscCall(DMPlexGetSubpointMap(*subdm, &map));
  PetscCall(DMLabelView(map, PETSC_VIEWER_STDOUT_WORLD));
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, subdm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(CreateMesh(PETSC_COMM_WORLD, &dm));
  PetscCall(CreateSubmesh(dm, PETSC_TRUE, &subdm));
  PetscCall(DMSetFromOptions(subdm));
  PetscCall(DMDestroy(&subdm));
  PetscCall(CreateSubmesh(dm, PETSC_FALSE, &subdm));
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
    args: -dm_coord_space 0 -dm_view ascii::ascii_info_detail -dm_plex_check_all

TEST*/
