static char help[] = "Reload a DMPLEX converted from a DMFOREST (p8est) from HDF5 in parallel and check its point SF.\n\
A DMPLEX converted from a DMFOREST orders its strata cells, faces, edges, vertices instead of cells, vertices, faces, edges.\n\n";

#include <petscdmplex.h>
#include <petscdmforest.h>
#include <petscviewerhdf5.h>
#include <petscsf.h>

int main(int argc, char **argv)
{
  DM          base, forest, plex, dm;
  PetscViewer viewer;
  PetscSF     sfXC;
  PetscInt    faces[3] = {4, 4, 4};

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  // Unrefined p8est forest over a box, converted to DMPLEX (strata ordered cells, faces, edges, vertices)
  PetscCall(DMPlexCreateBoxMesh(PETSC_COMM_WORLD, 3, PETSC_FALSE, faces, NULL, NULL, NULL, PETSC_TRUE, 0, PETSC_TRUE, &base));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &forest));
  PetscCall(DMSetType(forest, DMP8EST));
  PetscCall(DMForestSetBaseDM(forest, base));
  PetscCall(DMSetUp(forest));
  PetscCall(DMConvert(forest, DMPLEX, &plex));
  PetscCall(PetscObjectSetName((PetscObject)plex, "plex"));

  // Save the topology, with storage version 3.1.0 (the default 1.0.0 loader reorders the strata to cells, vertices, faces, edges)
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "forest_pointsf.h5", FILE_MODE_WRITE, &viewer));
  {
    struct _n_DMPlexStorageVersion version = {3, 1, 0};

    PetscCall(PetscViewerHDF5SetDMPlexStorageVersionWriting(viewer, &version));
  }
  PetscCall(DMPlexTopologyView(plex, viewer));
  PetscCall(PetscViewerDestroy(&viewer));

  // Load it back; the loaded DM has no tree, so DMPlexCheckPointSF() performs its depth check
  PetscCall(PetscViewerHDF5Open(PETSC_COMM_WORLD, "forest_pointsf.h5", FILE_MODE_READ, &viewer));
  PetscCall(DMCreate(PETSC_COMM_WORLD, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(PetscObjectSetName((PetscObject)dm, "plex"));
  PetscCall(DMPlexTopologyLoad(dm, viewer, &sfXC));
  PetscCall(PetscViewerDestroy(&viewer));

  // Without order-independent depth lookup this fails on 2+ ranks with "Leaf .. has depth 2 but remote (.., ..) depth is 3"
  PetscCall(DMPlexCheckPointSF(dm, NULL, PETSC_FALSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Point SF of the reloaded forest-ordered DMPLEX is consistent\n"));

  PetscCall(PetscSFDestroy(&sfXC));
  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&plex));
  PetscCall(DMDestroy(&forest));
  PetscCall(DMDestroy(&base));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: p4est hdf5

  test:
    suffix: 0
    nsize: 2

TEST*/
