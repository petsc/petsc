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

  ierr = DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 2, PETSC_FALSE, NULL, NULL, NULL, NULL, PETSC_TRUE, &base);CHKERRQ(ierr);
  ierr = DMSetFromOptions(base);CHKERRQ(ierr);

  ierr = DMCreate(PETSC_COMM_WORLD, &forest);CHKERRQ(ierr);
  ierr = DMSetType(forest, DMP4EST);CHKERRQ(ierr);
  ierr = DMForestSetBaseDM(forest, base);CHKERRQ(ierr);
  ierr = DMForestSetInitialRefinement(forest, min_refine);CHKERRQ(ierr);
  ierr = DMForestSetPartitionOverlap(forest, overlap);CHKERRQ(ierr);
  ierr = DMSetUp(forest);CHKERRQ(ierr);
  ierr = DMDestroy(&base);CHKERRQ(ierr);
  ierr = DMViewFromOptions(forest, NULL, "-dm_view");CHKERRQ(ierr);

  ierr = DMConvert(forest, DMPLEX, &plex);CHKERRQ(ierr);
  ierr = DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd);CHKERRQ(ierr);
  ierr = DMDestroy(&plex);CHKERRQ(ierr);
  ierr = PetscSectionCreate(PetscObjectComm((PetscObject) forest), &s);CHKERRQ(ierr);
  ierr = PetscSectionSetChart(s, vStart, vEnd);CHKERRQ(ierr);
  for (v = vStart; v < vEnd; ++v) {ierr = PetscSectionSetDof(s, v, 1);CHKERRQ(ierr);}
  ierr = PetscSectionSetUp(s);CHKERRQ(ierr);
  ierr = DMSetLocalSection(forest, s);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&s);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(forest, &g);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) g, "g");CHKERRQ(ierr);
  ierr = VecSet(g, 1.0);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "forest.h5", FILE_MODE_WRITE, &viewer);
  ierr = VecView(g, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(forest, &g2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) g2, "g");CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "forest.h5", FILE_MODE_READ, &viewer);
  ierr = VecLoad(g2, viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  /*  Check if the data is the same*/
  ierr = VecAXPY(g2, -1.0, g);CHKERRQ(ierr);
  ierr = VecNorm(g2, NORM_INFINITY, &diff);CHKERRQ(ierr);
  if (diff > PETSC_MACHINE_EPSILON) {ierr = PetscPrintf(PETSC_COMM_WORLD, "Check failed: %g\n", (double) diff);CHKERRQ(ierr);}

  ierr = VecDestroy(&g);CHKERRQ(ierr);
  ierr = VecDestroy(&g2);CHKERRQ(ierr);
  ierr = DMDestroy(&forest);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

  build:
    requires: hdf5 p4est

  test:
    args: -dm_plex_box_faces 3,3

TEST*/
