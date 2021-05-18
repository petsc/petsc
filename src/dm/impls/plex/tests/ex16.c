static char help[] = "Tests for creation of submeshes\n\n";

#include <petscdmplex.h>

PetscErrorCode CreateMesh(MPI_Comm comm, DM *dm)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, dm);CHKERRQ(ierr);
  ierr = DMSetType(*dm, DMPLEX);CHKERRQ(ierr);
  ierr = DMSetFromOptions(*dm);CHKERRQ(ierr);
  ierr = DMViewFromOptions(*dm, NULL, "-dm_view");CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

PetscErrorCode CreateSubmesh(DM dm, PetscBool start, DM *subdm)
{
  DMLabel        label, map;
  PetscInt       cStart, cEnd, cStartSub, cEndSub, c;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMPlexGetHeightStratum(dm, 0, &cStart, &cEnd);CHKERRQ(ierr);
  ierr = DMCreateLabel(dm, "cells");CHKERRQ(ierr);
  ierr = DMGetLabel(dm, "cells", &label);CHKERRQ(ierr);
  ierr = DMLabelClearStratum(label, 1);CHKERRQ(ierr);
  if (start) {cStartSub = cStart; cEndSub = cEnd/2;}
  else       {cStartSub = cEnd/2; cEndSub = cEnd;}
  for (c = cStartSub; c < cEndSub; ++c) {ierr = DMLabelSetValue(label, c, 1);CHKERRQ(ierr);}
  ierr = DMPlexFilter(dm, label, 1, subdm);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) *subdm, "Submesh");CHKERRQ(ierr);
  ierr = DMViewFromOptions(*subdm, NULL, "-dm_view");CHKERRQ(ierr);
  ierr = DMPlexGetSubpointMap(*subdm, &map);CHKERRQ(ierr);
  ierr = DMLabelView(map, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

int main(int argc, char **argv)
{
  DM             dm, subdm;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;
  ierr = CreateMesh(PETSC_COMM_WORLD, &dm);CHKERRQ(ierr);
  ierr = CreateSubmesh(dm, PETSC_TRUE, &subdm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(subdm);CHKERRQ(ierr);
  ierr = DMDestroy(&subdm);CHKERRQ(ierr);
  ierr = CreateSubmesh(dm, PETSC_FALSE, &subdm);CHKERRQ(ierr);
  ierr = DMSetFromOptions(subdm);CHKERRQ(ierr);
  ierr = DMDestroy(&subdm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  test:
    suffix: 0
    requires: triangle
    args: -dm_coord_space 0 -dm_view ascii::ascii_info_detail -dm_plex_check_all

TEST*/
