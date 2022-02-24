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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char*) 0, help);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  CHKERRMPI(MPI_Comm_rank(comm, &rank));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-n", &n, NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL, "-n_timesteps", &n_timesteps, NULL));
  PetscCheckFalse(n_timesteps < 0,comm, PETSC_ERR_USER_INPUT, "-n_timesteps must be nonnegative");

  /* create, initialize and write vectors */
  CHKERRQ(PetscRandomCreate(comm, &rand));
  CHKERRQ(PetscRandomSetFromOptions(rand));
  CHKERRQ(PetscViewerHDF5Open(comm, "ex19.h5", FILE_MODE_WRITE, &viewer));

  CHKERRQ(VecCreate(comm, &x1));
  CHKERRQ(PetscObjectSetName((PetscObject) x1, "x1"));
  CHKERRQ(VecSetSizes(x1, PETSC_DECIDE, n));
  CHKERRQ(VecSetFromOptions(x1));
  CHKERRQ(VecSetRandom(x1, rand));
  CHKERRQ(VecView(x1, viewer));

  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/testBlockSize"));
  CHKERRQ(VecCreate(comm, &x2));
  CHKERRQ(PetscObjectSetName((PetscObject) x2, "x2"));
  CHKERRQ(VecSetSizes(x2, PETSC_DECIDE, n));
  CHKERRQ(VecSetBlockSize(x2, 2));
  CHKERRQ(VecSetFromOptions(x2));
  CHKERRQ(VecSetRandom(x2, rand));
  CHKERRQ(VecView(x2, viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));

  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/testTimestep"));
  CHKERRQ(PetscViewerHDF5PushTimestepping(viewer));

  CHKERRQ(VecDuplicateVecs(x1, n_timesteps, &x3ts));
  for (i=0; i<n_timesteps; i++) {
    CHKERRQ(PetscObjectSetName((PetscObject) x3ts[i], "x3"));
    CHKERRQ(VecSetRandom(x3ts[i], rand));
    CHKERRQ(VecView(x3ts[i], viewer));
    CHKERRQ(PetscViewerHDF5IncrementTimestep(viewer));
  }

  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "testBlockSize"));
  CHKERRQ(VecDuplicateVecs(x2, n_timesteps, &x4ts));
  for (i=0; i<n_timesteps; i++) {
    CHKERRQ(PetscObjectSetName((PetscObject) x4ts[i], "x4"));
    CHKERRQ(VecSetRandom(x4ts[i], rand));
    CHKERRQ(PetscViewerHDF5SetTimestep(viewer, i));
    CHKERRQ(VecView(x4ts[i], viewer));
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));

  CHKERRQ(PetscViewerHDF5PopTimestepping(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));

  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(PetscRandomDestroy(&rand));

  /* read and compare */
  CHKERRQ(PetscViewerHDF5Open(comm, "ex19.h5", FILE_MODE_READ, &viewer));

  CHKERRQ(VecDuplicate(x1, &x1r));
  CHKERRQ(PetscObjectSetName((PetscObject) x1r, "x1"));
  CHKERRQ(VecLoad(x1r, viewer));
  CHKERRQ(VecEqual(x1, x1r, &equal));
  if (!equal) {
    CHKERRQ(VecView(x1, PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(VecView(x1r, PETSC_VIEWER_STDOUT_WORLD));
    SETERRQ(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x1 != x1r");
  }

  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/testBlockSize"));
  CHKERRQ(VecDuplicate(x2, &x2r));
  CHKERRQ(PetscObjectSetName((PetscObject) x2r, "x2"));
  CHKERRQ(VecLoad(x2r, viewer));
  CHKERRQ(VecEqual(x2, x2r, &equal));
  if (!equal) {
    CHKERRQ(VecView(x2, PETSC_VIEWER_STDOUT_WORLD));
    CHKERRQ(VecView(x2r, PETSC_VIEWER_STDOUT_WORLD));
    SETERRQ(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x2 != x2r");
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));

  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "/testTimestep"));
  CHKERRQ(PetscViewerHDF5PushTimestepping(viewer));

  CHKERRQ(VecDuplicate(x1, &x3r));
  CHKERRQ(PetscObjectSetName((PetscObject) x3r, "x3"));
  for (i=0; i<n_timesteps; i++) {
    CHKERRQ(PetscViewerHDF5SetTimestep(viewer, i));
    CHKERRQ(VecLoad(x3r, viewer));
    CHKERRQ(VecEqual(x3r, x3ts[i], &equal));
    if (!equal) {
      CHKERRQ(VecView(x3r, PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(VecView(x3ts[i], PETSC_VIEWER_STDOUT_WORLD));
      SETERRQ(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x3ts[%" PetscInt_FMT "] != x3r", i);
    }
  }

  CHKERRQ(PetscViewerHDF5PushGroup(viewer, "testBlockSize"));
  CHKERRQ(VecDuplicate(x2, &x4r));
  CHKERRQ(PetscObjectSetName((PetscObject) x4r, "x4"));
  CHKERRQ(PetscViewerHDF5SetTimestep(viewer, 0));
  for (i=0; i<n_timesteps; i++) {
    CHKERRQ(VecLoad(x4r, viewer));
    CHKERRQ(VecEqual(x4r, x4ts[i], &equal));
    if (!equal) {
      CHKERRQ(VecView(x4r, PETSC_VIEWER_STDOUT_WORLD));
      CHKERRQ(VecView(x4ts[i], PETSC_VIEWER_STDOUT_WORLD));
      SETERRQ(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x4ts[%" PetscInt_FMT "] != x4r", i);
    }
    CHKERRQ(PetscViewerHDF5IncrementTimestep(viewer));
  }
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));

  CHKERRQ(PetscViewerHDF5PopTimestepping(viewer));
  CHKERRQ(PetscViewerHDF5PopGroup(viewer));

  /* cleanup */
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&x1));
  CHKERRQ(VecDestroy(&x2));
  CHKERRQ(VecDestroyVecs(n_timesteps, &x3ts));
  CHKERRQ(VecDestroyVecs(n_timesteps, &x4ts));
  CHKERRQ(VecDestroy(&x1r));
  CHKERRQ(VecDestroy(&x2r));
  CHKERRQ(VecDestroy(&x3r));
  CHKERRQ(VecDestroy(&x4r));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     build:
       requires: hdf5

     test:
       nsize: {{1 2 3 4}}

TEST*/
