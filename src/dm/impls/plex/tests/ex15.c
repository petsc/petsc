static char help[] = "An example of writing a global Vec from a DMPlex with HDF5 format.\n";

#include <petscdmplex.h>
#include <petscviewerhdf5.h>

int main(int argc, char **argv)
{
  MPI_Comm     comm;
  DM           dm;
  Vec          v, nv, rv, coord;
  PetscBool    test_read = PETSC_FALSE, verbose = PETSC_FALSE, flg;
  PetscViewer  hdf5Viewer;
  PetscInt     numFields   = 1;
  PetscInt     numBC       = 0;
  PetscInt     numComp[1]  = {2};
  PetscInt     numDof[3]   = {2, 0, 0};
  PetscInt     bcFields[1] = {0};
  IS           bcPoints[1] = {NULL};
  PetscSection section;
  PetscReal    norm;
  PetscInt     dim;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  comm = PETSC_COMM_WORLD;

  PetscOptionsBegin(PETSC_COMM_WORLD, "", "Test Options", "none");
  PetscCall(PetscOptionsBool("-test_read", "Test reading from the HDF5 file", "", PETSC_FALSE, &test_read, NULL));
  PetscCall(PetscOptionsBool("-verbose", "print the Vecs", "", PETSC_FALSE, &verbose, NULL));
  PetscOptionsEnd();

  PetscCall(DMCreate(comm, &dm));
  PetscCall(DMSetType(dm, DMPLEX));
  PetscCall(DMPlexDistributeSetDefault(dm, PETSC_FALSE));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMViewFromOptions(dm, NULL, "-dm_view"));
  PetscCall(DMGetDimension(dm, &dim));
  numDof[0] = dim;
  PetscCall(DMSetNumFields(dm, numFields));
  PetscCall(DMPlexCreateSection(dm, NULL, numComp, numDof, numBC, bcFields, bcPoints, NULL, NULL, &section));
  PetscCall(DMSetLocalSection(dm, section));
  PetscCall(PetscSectionDestroy(&section));
  PetscCall(DMSetUseNatural(dm, PETSC_TRUE));
  {
    PetscPartitioner part;
    DM               dmDist;

    PetscCall(DMPlexGetPartitioner(dm, &part));
    PetscCall(PetscPartitionerSetFromOptions(part));
    PetscCall(DMPlexDistribute(dm, 0, NULL, &dmDist));
    if (dmDist) {
      PetscCall(DMDestroy(&dm));
      dm = dmDist;
    }
  }

  PetscCall(DMCreateGlobalVector(dm, &v));
  PetscCall(PetscObjectSetName((PetscObject)v, "V"));
  PetscCall(DMGetCoordinates(dm, &coord));
  PetscCall(VecCopy(coord, v));

  if (verbose) {
    PetscInt size, bs;

    PetscCall(VecGetSize(v, &size));
    PetscCall(VecGetBlockSize(v, &bs));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==== original V in global ordering. size==%d\tblock size=%d\n", size, bs));
    PetscCall(VecView(v, PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(DMPlexCreateNaturalVector(dm, &nv));
  PetscCall(PetscObjectSetName((PetscObject)nv, "NV"));
  PetscCall(DMPlexGlobalToNaturalBegin(dm, v, nv));
  PetscCall(DMPlexGlobalToNaturalEnd(dm, v, nv));

  if (verbose) {
    PetscInt size, bs;

    PetscCall(VecGetSize(nv, &size));
    PetscCall(VecGetBlockSize(nv, &bs));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "====  V in natural ordering. size==%d\tblock size=%d\n", size, bs));
    PetscCall(VecView(nv, PETSC_VIEWER_STDOUT_WORLD));
  }

  PetscCall(VecViewFromOptions(v, NULL, "-global_vec_view"));

  if (test_read) {
    PetscCall(DMCreateGlobalVector(dm, &rv));
    PetscCall(PetscObjectSetName((PetscObject)rv, "V"));
    /* Test native read */
    PetscCall(PetscViewerHDF5Open(comm, "V.h5", FILE_MODE_READ, &hdf5Viewer));
    PetscCall(PetscViewerPushFormat(hdf5Viewer, PETSC_VIEWER_NATIVE));
    PetscCall(VecLoad(rv, hdf5Viewer));
    PetscCall(PetscViewerPopFormat(hdf5Viewer));
    PetscCall(PetscViewerDestroy(&hdf5Viewer));
    if (verbose) {
      PetscInt size, bs;

      PetscCall(VecGetSize(rv, &size));
      PetscCall(VecGetBlockSize(rv, &bs));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==== Vector from file. size==%d\tblock size=%d\n", size, bs));
      PetscCall(VecView(rv, PETSC_VIEWER_STDOUT_WORLD));
    }
    PetscCall(VecEqual(rv, v, &flg));
    if (flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "V and RV are equal\n"));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "V and RV are not equal\n\n"));
      PetscCall(VecAXPY(rv, -1.0, v));
      PetscCall(VecNorm(rv, NORM_INFINITY, &norm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "diff norm is = %g\n", (double)norm));
      PetscCall(VecView(rv, PETSC_VIEWER_STDOUT_WORLD));
    }
    /* Test raw read */
    PetscCall(PetscViewerHDF5Open(comm, "V.h5", FILE_MODE_READ, &hdf5Viewer));
    PetscCall(VecLoad(rv, hdf5Viewer));
    PetscCall(PetscViewerDestroy(&hdf5Viewer));
    if (verbose) {
      PetscInt size, bs;

      PetscCall(VecGetSize(rv, &size));
      PetscCall(VecGetBlockSize(rv, &bs));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "==== Vector from file. size==%d\tblock size=%d\n", size, bs));
      PetscCall(VecView(rv, PETSC_VIEWER_STDOUT_WORLD));
    }
    PetscCall(VecEqual(rv, nv, &flg));
    if (flg) {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NV and RV are equal\n"));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NV and RV are not equal\n\n"));
      PetscCall(VecAXPY(rv, -1.0, v));
      PetscCall(VecNorm(rv, NORM_INFINITY, &norm));
      PetscCall(PetscPrintf(PETSC_COMM_WORLD, "diff norm is = %g\n", (double)norm));
      PetscCall(VecView(rv, PETSC_VIEWER_STDOUT_WORLD));
    }
    PetscCall(VecDestroy(&rv));
  }
  PetscCall(VecDestroy(&nv));
  PetscCall(VecDestroy(&v));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
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
