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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  PetscCall(DMCreate(PETSC_COMM_WORLD, &base));
  PetscCall(DMSetType(base, DMPLEX));
  PetscCall(DMSetFromOptions(base));

  PetscCall(DMCreate(PETSC_COMM_WORLD, &forest));
  PetscCall(DMSetType(forest, DMP4EST));
  PetscCall(DMForestSetBaseDM(forest, base));
  PetscCall(DMForestSetInitialRefinement(forest, min_refine));
  PetscCall(DMForestSetPartitionOverlap(forest, overlap));
  PetscCall(DMSetUp(forest));
  PetscCall(DMDestroy(&base));
  PetscCall(DMViewFromOptions(forest, NULL, "-dm_view"));

  PetscCall(DMConvert(forest, DMPLEX, &plex));
  PetscCall(DMPlexGetDepthStratum(plex, 0, &vStart, &vEnd));
  PetscCall(DMDestroy(&plex));
  PetscCall(PetscSectionCreate(PetscObjectComm((PetscObject) forest), &s));
  PetscCall(PetscSectionSetChart(s, vStart, vEnd));
  for (v = vStart; v < vEnd; ++v) PetscCall(PetscSectionSetDof(s, v, 1));
  PetscCall(PetscSectionSetUp(s));
  PetscCall(DMSetLocalSection(forest, s));
  PetscCall(PetscSectionDestroy(&s));

  PetscCall(DMCreateGlobalVector(forest, &g));
  PetscCall(PetscObjectSetName((PetscObject) g, "g"));
  PetscCall(VecSet(g, 1.0));
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "forest.h5", FILE_MODE_WRITE, &viewer));
  PetscCall(VecView(g, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  PetscCall(DMCreateGlobalVector(forest, &g2));
  PetscCall(PetscObjectSetName((PetscObject) g2, "g"));
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "forest.h5", FILE_MODE_READ, &viewer));
  PetscCall(VecLoad(g2, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  /*  Check if the data is the same*/
  PetscCall(VecAXPY(g2, -1.0, g));
  PetscCall(VecNorm(g2, NORM_INFINITY, &diff));
  if (diff > PETSC_MACHINE_EPSILON) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Check failed: %g\n", (double) diff));

  PetscCall(VecDestroy(&g));
  PetscCall(VecDestroy(&g2));
  PetscCall(DMDestroy(&forest));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: hdf5 p4est

  test:
    args: -dm_plex_simplex 0 -dm_plex_box_faces 3,3

TEST*/
