static char help[] = "Parallel HDF5 Vec Viewing.\n\n";

/*T
   Concepts: vectors^viewing
   Concepts: viewers^hdf5
   Processors: n
T*/

#include <petscvec.h>
#include <petscviewerhdf5.h>

int main(int argc,char **argv)
{
  Vec            x1, x2, *x3ts, *x4ts;
  Vec            x1r, x2r, x3r, x4r;
  PetscViewer    viewer;
  PetscRandom    rand;
  PetscMPIInt    rank;
  PetscInt       i, n = 6, n_timesteps = 5;
  PetscBool      equal;
  MPI_Comm       comm;

  PetscFunctionBegin;
  PetscCall(PetscInitialize(&argc, &argv, (char*) 0, help));
  comm = PETSC_COMM_WORLD;
  PetscCallMPI(MPI_Comm_rank(comm, &rank));
  PetscCall(PetscOptionsGetInt(NULL,NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL, "-n_timesteps", &n_timesteps, NULL));
  PetscCheck(n_timesteps >= 0,comm, PETSC_ERR_USER_INPUT, "-n_timesteps must be nonnegative");

  /* create, initialize and write vectors */
  PetscCall(PetscRandomCreate(comm, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(PetscViewerHDF5Open(comm, "ex19.h5", FILE_MODE_WRITE, &viewer));

  PetscCall(VecCreate(comm, &x1));
  PetscCall(PetscObjectSetName((PetscObject) x1, "x1"));
  PetscCall(VecSetSizes(x1, PETSC_DECIDE, n));
  PetscCall(VecSetFromOptions(x1));
  PetscCall(VecSetRandom(x1, rand));
  PetscCall(VecView(x1, viewer));

  PetscCall(PetscViewerHDF5PushGroup(viewer, "/testBlockSize"));
  PetscCall(VecCreate(comm, &x2));
  PetscCall(PetscObjectSetName((PetscObject) x2, "x2"));
  PetscCall(VecSetSizes(x2, PETSC_DECIDE, n));
  PetscCall(VecSetBlockSize(x2, 2));
  PetscCall(VecSetFromOptions(x2));
  PetscCall(VecSetRandom(x2, rand));
  PetscCall(VecView(x2, viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  PetscCall(PetscViewerHDF5PushGroup(viewer, "/testTimestep"));
  PetscCall(PetscViewerHDF5PushTimestepping(viewer));

  PetscCall(VecDuplicateVecs(x1, n_timesteps, &x3ts));
  for (i=0; i<n_timesteps; i++) {
    PetscCall(PetscObjectSetName((PetscObject) x3ts[i], "x3"));
    PetscCall(VecSetRandom(x3ts[i], rand));
    PetscCall(VecView(x3ts[i], viewer));
    PetscCall(PetscViewerHDF5IncrementTimestep(viewer));
  }

  PetscCall(PetscViewerHDF5PushGroup(viewer, "testBlockSize"));
  PetscCall(VecDuplicateVecs(x2, n_timesteps, &x4ts));
  for (i=0; i<n_timesteps; i++) {
    PetscCall(PetscObjectSetName((PetscObject) x4ts[i], "x4"));
    PetscCall(VecSetRandom(x4ts[i], rand));
    PetscCall(PetscViewerHDF5SetTimestep(viewer, i));
    PetscCall(VecView(x4ts[i], viewer));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  PetscCall(PetscViewerHDF5PopTimestepping(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(PetscRandomDestroy(&rand));

  /* read and compare */
  PetscCall(PetscViewerHDF5Open(comm, "ex19.h5", FILE_MODE_READ, &viewer));

  PetscCall(VecDuplicate(x1, &x1r));
  PetscCall(PetscObjectSetName((PetscObject) x1r, "x1"));
  PetscCall(VecLoad(x1r, viewer));
  PetscCall(VecEqual(x1, x1r, &equal));
  if (!equal) {
    PetscCall(VecView(x1, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecView(x1r, PETSC_VIEWER_STDOUT_WORLD));
    SETERRQ(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x1 != x1r");
  }

  PetscCall(PetscViewerHDF5PushGroup(viewer, "/testBlockSize"));
  PetscCall(VecDuplicate(x2, &x2r));
  PetscCall(PetscObjectSetName((PetscObject) x2r, "x2"));
  PetscCall(VecLoad(x2r, viewer));
  PetscCall(VecEqual(x2, x2r, &equal));
  if (!equal) {
    PetscCall(VecView(x2, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(VecView(x2r, PETSC_VIEWER_STDOUT_WORLD));
    SETERRQ(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x2 != x2r");
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  PetscCall(PetscViewerHDF5PushGroup(viewer, "/testTimestep"));
  PetscCall(PetscViewerHDF5PushTimestepping(viewer));

  PetscCall(VecDuplicate(x1, &x3r));
  PetscCall(PetscObjectSetName((PetscObject) x3r, "x3"));
  for (i=0; i<n_timesteps; i++) {
    PetscCall(PetscViewerHDF5SetTimestep(viewer, i));
    PetscCall(VecLoad(x3r, viewer));
    PetscCall(VecEqual(x3r, x3ts[i], &equal));
    if (!equal) {
      PetscCall(VecView(x3r, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(VecView(x3ts[i], PETSC_VIEWER_STDOUT_WORLD));
      SETERRQ(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x3ts[%" PetscInt_FMT "] != x3r", i);
    }
  }

  PetscCall(PetscViewerHDF5PushGroup(viewer, "testBlockSize"));
  PetscCall(VecDuplicate(x2, &x4r));
  PetscCall(PetscObjectSetName((PetscObject) x4r, "x4"));
  PetscCall(PetscViewerHDF5SetTimestep(viewer, 0));
  for (i=0; i<n_timesteps; i++) {
    PetscCall(VecLoad(x4r, viewer));
    PetscCall(VecEqual(x4r, x4ts[i], &equal));
    if (!equal) {
      PetscCall(VecView(x4r, PETSC_VIEWER_STDOUT_WORLD));
      PetscCall(VecView(x4ts[i], PETSC_VIEWER_STDOUT_WORLD));
      SETERRQ(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x4ts[%" PetscInt_FMT "] != x4r", i);
    }
    PetscCall(PetscViewerHDF5IncrementTimestep(viewer));
  }
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  PetscCall(PetscViewerHDF5PopTimestepping(viewer));
  PetscCall(PetscViewerHDF5PopGroup(viewer));

  /* cleanup */
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroyVecs(n_timesteps, &x3ts));
  PetscCall(VecDestroyVecs(n_timesteps, &x4ts));
  PetscCall(VecDestroy(&x1r));
  PetscCall(VecDestroy(&x2r));
  PetscCall(VecDestroy(&x3r));
  PetscCall(VecDestroy(&x4r));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

     build:
       requires: hdf5

     test:
       nsize: {{1 2 3 4}}

TEST*/
