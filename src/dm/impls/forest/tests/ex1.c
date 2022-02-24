static char help[] = "Test HDF5 input and output.\n\
This exposed a bug with sharing discretizations.\n\n\n";

#include <petscdmforest.h>
#include <petscdmplex.h>
#include <petscviewerhdf5.h>

int main (int argc, char **argv)
{

  DM             base, forest, plex;
  Vec            g, g2;
  PetscSection   s;
  PetscViewer    viewer;
  PetscReal      diff;
  PetscInt       min_refine = 2, overlap = 0;
  PetscInt       vStart, vEnd, v;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, NULL, help);if (ierr) return ierr;

  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &base));
  CHKERRQ(DMSetType(base, DMPLEX));
  CHKERRQ(DMSetFromOptions(base));

  CHKERRQ(DMCreate(PETSC_COMM_WORLD, &forest));
  CHKERRQ(DMSetType(forest, DMP4EST));
  CHKERRQ(DMForestSetBaseDM(forest, base));
  CHKERRQ(DMForestSetInitialRefinement(forest, min_refine));
  CHKERRQ(DMForestSetPartitionOverlap(forest, overlap));
  CHKERRQ(DMSetUp(forest));
  CHKERRQ(DMDestroy(&base));
  CHKERRQ(DMViewFromOptions(forest, NULL, "-dm_view"));

  CHKERRQ(DMConvert(forest, DMPLEX, &plex));
  CHKERRQ(DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd));
  CHKERRQ(DMDestroy(&plex));
  CHKERRQ(PetscSectionCreate(PetscObjectComm((PetscObject) forest), &s));
  CHKERRQ(PetscSectionSetChart(s, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) CHKERRQ(PetscSectionSetDof(s, v, 1));
  CHKERRQ(PetscSectionSetUp(s));
  CHKERRQ(DMSetLocalSection(forest, s));
  CHKERRQ(PetscSectionDestroy(&s));

  CHKERRQ(DMCreateGlobalVector(forest, &g));
  CHKERRQ(PetscObjectSetName((PetscObject) g, "g"));
  CHKERRQ(VecSet(g, 1.0));
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD, "forest.h5", FILE_MODE_WRITE, &viewer));
  CHKERRQ(VecView(g, viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  CHKERRQ(DMCreateGlobalVector(forest, &g2));
  CHKERRQ(PetscObjectSetName((PetscObject) g2, "g"));
  CHKERRQ(PetscViewerHDF5Open(PETSC_COMM_WORLD, "forest.h5", FILE_MODE_READ, &viewer));
  CHKERRQ(VecLoad(g2, viewer));
  CHKERRQ(PetscViewerDestroy(&viewer));

  /*  Check if the data is the same*/
  CHKERRQ(VecAXPY(g2, -1.0, g));
  CHKERRQ(VecNorm(g2, NORM_INFINITY, &diff));
  if (diff > PETSC_MACHINE_EPSILON) CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "Check failed: %g\n", (double) diff));

  CHKERRQ(VecDestroy(&g));
  CHKERRQ(VecDestroy(&g2));
  CHKERRQ(DMDestroy(&forest));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: hdf5 p4est

  test:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,3

TEST*/
