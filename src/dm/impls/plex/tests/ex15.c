static char help[] = "An example of writing a global Vec from a DMPlex with HDF5 format.\n";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>

int main(int argc, char **argv)
{
  MPI_Comm       comm;
  DM             dm;
  Vec            v, nv, rv, coord;
  PetscBool      test_read = PETSC_FALSE, verbose = PETSC_FALSE, flg;
  PetscViewer    hdf5Viewer;
  PetscInt       numFields   = 1;
  PetscInt       numBC       = 0;
  PetscInt       numComp[1]  = {2};
  PetscInt       numDof[3]   = {2, 0, 0};
  PetscInt       bcFields[1] = {0};
  IS             bcPoints[1] = {NULL};
  PetscSection   section;
  PetscReal      norm;
  PetscInt       dim;
  PetscErrorCode ierr;

  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD,"","Test Options","none");CHKERRQ(ierr);
  CHKERRQ(PetscOptionsBool("-test_read","Test reading from the HDF5 file","",PETSC_FALSE,&test_read,NULL));
  CHKERRQ(PetscOptionsBool("-verbose","print the Vecs","",PETSC_FALSE,&verbose,NULL));
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  CHKERRQ(DMCreate(comm, &dm));
  CHKERRQ(DMSetType(dm, DMPLEX));
  CHKERRQ(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMViewFromOptions(dm, NULL, "-dm_view"));
  CHKERRQ(DMGetDimension(dm, &dim));
  numDof[0] = dim;
  CHKERRQ(DMSetNumFields(dm, numFields));
  CHKERRQ(DMPlexCreateSection(dm, NULL, numComp, numDof, numBC, bcFields, bcPoints, NULL, NULL, &section));
  CHKERRQ(DMSetLocalSection(dm, section));
  CHKERRQ(PetscSectionDestroy(&section));
  CHKERRQ(DMSetUseNatural(dm, PETSC_TRUE));
  {
    PetscPartitioner part;
    DM               dmDist;

    CHKERRQ(DMPlexGetPartitioner(dm,&part));
    CHKERRQ(PetscPartitionerSetFromOptions(part));
    CHKERRQ(DMPlexDistribute(dm, 0, NULL, &dmDist));
    if (dmDist) {
      CHKERRQ(DMDestroy(&dm));
      dm   = dmDist;
    }
  }

  CHKERRQ(DMCreateGlobalVector(dm, &v));
  CHKERRQ(PetscObjectSetName((PetscObject) v, "V"));
  CHKERRQ(DMGetCoordinates(dm, &coord));
  CHKERRQ(VecCopy(coord, v));

  if (verbose) {
    PetscInt size, bs;

    CHKERRQ(VecGetSize(v, &size));
    CHKERRQ(VecGetBlockSize(v, &bs));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "==== original V in global ordering. size==%d\tblock size=%d\n", size, bs));
    CHKERRQ(VecView(v, PETSC_VIEWER_STDOUT_WORLD));
  }

  CHKERRQ(DMCreateGlobalVector(dm, &nv));
  CHKERRQ(PetscObjectSetName((PetscObject) nv, "NV"));
  CHKERRQ(DMPlexGlobalToNaturalBegin(dm, v, nv));
  CHKERRQ(DMPlexGlobalToNaturalEnd(dm, v, nv));

  if (verbose) {
    PetscInt size, bs;

    CHKERRQ(VecGetSize(nv, &size));
    CHKERRQ(VecGetBlockSize(nv, &bs));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "====  V in natural ordering. size==%d\tblock size=%d\n", size, bs));
    CHKERRQ(VecView(nv, PETSC_VIEWER_STDOUT_WORLD));
  }

  CHKERRQ(VecViewFromOptions(v, NULL, "-global_vec_view"));

  if (test_read) {
    CHKERRQ(DMCreateGlobalVector(dm, &rv));
    CHKERRQ(PetscObjectSetName((PetscObject) rv, "V"));
    /* Test native read */
    CHKERRQ(PetscViewerHDF5Open(comm, "V.h5", FILE_MODE_READ, &hdf5Viewer));
    CHKERRQ(PetscViewerPushFormat(hdf5Viewer, PETSC_VIEWER_NATIVE));
    CHKERRQ(VecLoad(rv, hdf5Viewer));
    CHKERRQ(PetscViewerPopFormat(hdf5Viewer));
    CHKERRQ(PetscViewerDestroy(&hdf5Viewer));
    if (verbose) {
      PetscInt size, bs;

      CHKERRQ(VecGetSize(rv, &size));
      CHKERRQ(VecGetBlockSize(rv, &bs));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "==== Vector from file. size==%d\tblock size=%d\n", size, bs));
      CHKERRQ(VecView(rv, PETSC_VIEWER_STDOUT_WORLD));
    }
    CHKERRQ(VecEqual(rv, v, &flg));
    if (flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "V and RV are equal\n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "V and RV are not equal\n\n"));
      CHKERRQ(VecAXPY(rv, -1.0, v));
      CHKERRQ(VecNorm(rv, NORM_INFINITY, &norm));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "diff norm is = %g\n", (double) norm));
      CHKERRQ(VecView(rv, PETSC_VIEWER_STDOUT_WORLD));
    }
    /* Test raw read */
    CHKERRQ(PetscViewerHDF5Open(comm, "V.h5", FILE_MODE_READ, &hdf5Viewer));
    CHKERRQ(VecLoad(rv, hdf5Viewer));
    CHKERRQ(PetscViewerDestroy(&hdf5Viewer));
    if (verbose) {
      PetscInt size, bs;

      CHKERRQ(VecGetSize(rv, &size));
      CHKERRQ(VecGetBlockSize(rv, &bs));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "==== Vector from file. size==%d\tblock size=%d\n", size, bs));
      CHKERRQ(VecView(rv, PETSC_VIEWER_STDOUT_WORLD));
    }
    CHKERRQ(VecEqual(rv, nv, &flg));
    if (flg) {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "NV and RV are equal\n"));
    } else {
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "NV and RV are not equal\n\n"));
      CHKERRQ(VecAXPY(rv, -1.0, v));
      CHKERRQ(VecNorm(rv, NORM_INFINITY, &norm));
      CHKERRQ(PetscPrintf(PETSC_COMM_WORLD, "diff norm is = %g\n", (double) norm));
      CHKERRQ(VecView(rv, PETSC_VIEWER_STDOUT_WORLD));
    }
    CHKERRQ(VecDestroy(&rv));
  }
  CHKERRQ(VecDestroy(&nv));
  CHKERRQ(VecDestroy(&v));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST
  build:
    requires: triangle hdf5
  test:
    suffix: 0
    requires: triangle hdf5
    nsize: 2
    args: -petscpartitioner_type simple -verbose -globaltonatural_sf_view
  test:
    suffix: 1
    requires: triangle hdf5
    nsize: 2
    args: -petscpartitioner_type simple -verbose -global_vec_view hdf5:V.h5:native -test_read

TEST*/
