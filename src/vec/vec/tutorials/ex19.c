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
  ierr = MPI_Comm_rank(comm, &rank);CHKERRMPI(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-n", &n, NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-n_timesteps", &n_timesteps, NULL);CHKERRQ(ierr);
  if (n_timesteps < 0) SETERRQ(comm, PETSC_ERR_USER_INPUT, "-n_timesteps must be nonnegative");

  /* create, initialize and write vectors */
  ierr = PetscRandomCreate(comm, &rand);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rand);CHKERRQ(ierr);
  ierr = PetscViewerHDF5Open(comm, "ex19.h5", FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);

  ierr = VecCreate(comm, &x1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x1, "x1");CHKERRQ(ierr);
  ierr = VecSetSizes(x1, PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x1);CHKERRQ(ierr);
  ierr = VecSetRandom(x1, rand);CHKERRQ(ierr);
  ierr = VecView(x1, viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/testBlockSize");CHKERRQ(ierr);
  ierr = VecCreate(comm, &x2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x2, "x2");CHKERRQ(ierr);
  ierr = VecSetSizes(x2, PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(x2, 2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x2);CHKERRQ(ierr);
  ierr = VecSetRandom(x2, rand);CHKERRQ(ierr);
  ierr = VecView(x2, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/testTimestep");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);

  ierr = VecDuplicateVecs(x1, n_timesteps, &x3ts);CHKERRQ(ierr);
  for (i=0; i<n_timesteps; i++) {
    ierr = PetscObjectSetName((PetscObject) x3ts[i], "x3");CHKERRQ(ierr);
    ierr = VecSetRandom(x3ts[i], rand);CHKERRQ(ierr);
    ierr = VecView(x3ts[i], viewer);CHKERRQ(ierr);
    ierr = PetscViewerHDF5IncrementTimestep(viewer);CHKERRQ(ierr);
  }

  ierr = PetscViewerHDF5PushGroup(viewer, "testBlockSize");CHKERRQ(ierr);
  ierr = VecDuplicateVecs(x2, n_timesteps, &x4ts);CHKERRQ(ierr);
  for (i=0; i<n_timesteps; i++) {
    ierr = PetscObjectSetName((PetscObject) x4ts[i], "x4");CHKERRQ(ierr);
    ierr = VecSetRandom(x4ts[i], rand);CHKERRQ(ierr);
    ierr = PetscViewerHDF5SetTimestep(viewer, i);CHKERRQ(ierr);
    ierr = VecView(x4ts[i], viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rand);CHKERRQ(ierr);

  /* read and compare */
  ierr = PetscViewerHDF5Open(comm, "ex19.h5", FILE_MODE_READ, &viewer);CHKERRQ(ierr);

  ierr = VecDuplicate(x1, &x1r);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x1r, "x1");CHKERRQ(ierr);
  ierr = VecLoad(x1r, viewer);CHKERRQ(ierr);
  ierr = VecEqual(x1, x1r, &equal);CHKERRQ(ierr);
  if (!equal) {
    ierr = VecView(x1, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecView(x1r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    SETERRQ(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x1 != x1r");
  }

  ierr = PetscViewerHDF5PushGroup(viewer, "/testBlockSize");CHKERRQ(ierr);
  ierr = VecDuplicate(x2, &x2r);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x2r, "x2");CHKERRQ(ierr);
  ierr = VecLoad(x2r, viewer);CHKERRQ(ierr);
  ierr = VecEqual(x2, x2r, &equal);CHKERRQ(ierr);
  if (!equal) {
    ierr = VecView(x2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecView(x2r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    SETERRQ(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x2 != x2r");
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/testTimestep");CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushTimestepping(viewer);CHKERRQ(ierr);

  ierr = VecDuplicate(x1, &x3r);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x3r, "x3");CHKERRQ(ierr);
  for (i=0; i<n_timesteps; i++) {
    ierr = PetscViewerHDF5SetTimestep(viewer, i);CHKERRQ(ierr);
    ierr = VecLoad(x3r, viewer);CHKERRQ(ierr);
    ierr = VecEqual(x3r, x3ts[i], &equal);CHKERRQ(ierr);
    if (!equal) {
      ierr = VecView(x3r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = VecView(x3ts[i], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      SETERRQ1(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x3ts[%" PetscInt_FMT "] != x3r", i);
    }
  }

  ierr = PetscViewerHDF5PushGroup(viewer, "testBlockSize");CHKERRQ(ierr);
  ierr = VecDuplicate(x2, &x4r);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x4r, "x4");CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, 0);CHKERRQ(ierr);
  for (i=0; i<n_timesteps; i++) {
    ierr = VecLoad(x4r, viewer);CHKERRQ(ierr);
    ierr = VecEqual(x4r, x4ts[i], &equal);CHKERRQ(ierr);
    if (!equal) {
      ierr = VecView(x4r, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      ierr = VecView(x4ts[i], PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
      SETERRQ1(comm, PETSC_ERR_PLIB, "Error in HDF5 viewer: x4ts[%" PetscInt_FMT "] != x4r", i);
    }
    ierr = PetscViewerHDF5IncrementTimestep(viewer);CHKERRQ(ierr);
  }
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PopTimestepping(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);

  /* cleanup */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&x1);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n_timesteps, &x3ts);CHKERRQ(ierr);
  ierr = VecDestroyVecs(n_timesteps, &x4ts);CHKERRQ(ierr);
  ierr = VecDestroy(&x1r);CHKERRQ(ierr);
  ierr = VecDestroy(&x2r);CHKERRQ(ierr);
  ierr = VecDestroy(&x3r);CHKERRQ(ierr);
  ierr = VecDestroy(&x4r);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     build:
       requires: hdf5

     test:
       nsize: {{1 2 3 4}}

TEST*/
