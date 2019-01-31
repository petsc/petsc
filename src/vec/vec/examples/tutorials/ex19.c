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
  Vec            x1, x2, y1, y2, y3, y4;
  PetscViewer    viewer;
  PetscMPIInt    rank;
  PetscInt       i, nlocal, n = 6;
  PetscScalar    *array;
  PetscBool      equal;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char*) 0, help);if (ierr) return ierr;
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL, "-n", &n, NULL);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &x1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x1, "TestVec");CHKERRQ(ierr);
  ierr = VecSetSizes(x1, PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x1);CHKERRQ(ierr);

  /* initialize x1 */
  ierr = VecGetLocalSize(x1, &nlocal);CHKERRQ(ierr);
  ierr = VecGetArray(x1, &array);CHKERRQ(ierr);
  for (i = 0; i < nlocal; i++) array[i] = rank + 1;
  ierr = VecRestoreArray(x1, &array);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &x2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) x2, "TestVec2");CHKERRQ(ierr);
  ierr = VecSetSizes(x2, PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(x2, 2);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x2);CHKERRQ(ierr);

  /* initialize x2 */
  ierr = VecGetLocalSize(x2, &nlocal);CHKERRQ(ierr);
  ierr = VecGetArray(x2, &array);CHKERRQ(ierr);
  for (i = 0; i < nlocal; i++) array[i] = rank + 1;
  ierr = VecRestoreArray(x2, &array);CHKERRQ(ierr);

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "ex19.h5", FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
  ierr = VecView(x1, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/testBlockSize");CHKERRQ(ierr);
  ierr = VecView(x2, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/testTimestep");CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, 0);CHKERRQ(ierr);
  ierr = VecView(x2, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, 1);CHKERRQ(ierr);
  ierr = VecView(x2, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD, &y1);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) y1, "TestVec");CHKERRQ(ierr);
  ierr = VecSetSizes(y1, PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y1);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y2);CHKERRQ(ierr);
  ierr = VecSetBlockSize(y2, 2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) y2, "TestVec2");CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y3);CHKERRQ(ierr);
  ierr = VecSetBlockSize(y3, 2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) y3, "TestVec2");CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&y4);CHKERRQ(ierr);
  ierr = VecSetBlockSize(y4, 2);CHKERRQ(ierr);
  ierr = PetscObjectSetName((PetscObject) y4, "TestVec2");CHKERRQ(ierr);

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "ex19.h5", FILE_MODE_READ, &viewer);CHKERRQ(ierr);
  ierr = VecLoad(y1, viewer);CHKERRQ(ierr);

  ierr = PetscViewerHDF5PushGroup(viewer, "/testBlockSize");CHKERRQ(ierr);
  ierr = VecLoad(y2, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PushGroup(viewer, "/testTimestep");CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, 0);CHKERRQ(ierr);
  ierr = VecLoad(y3, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5SetTimestep(viewer, 1);CHKERRQ(ierr);
  ierr = VecLoad(y4, viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerHDF5PopGroup(viewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);

  ierr = VecEqual(x1, y1, &equal);CHKERRQ(ierr);
  if (!equal) {
    ierr = VecView(x1, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecView(y1, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Error in HDF5 viewer");
  }
  ierr = VecEqual(x2, y2, &equal);CHKERRQ(ierr);
  if (!equal) {
    ierr = VecView(x2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    ierr = VecView(y2, PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
    SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB, "Error in HDF5 viewer");
  }

  ierr = VecDestroy(&x1);CHKERRQ(ierr);
  ierr = VecDestroy(&x2);CHKERRQ(ierr);
  ierr = VecDestroy(&y1);CHKERRQ(ierr);
  ierr = VecDestroy(&y2);CHKERRQ(ierr);
  ierr = VecDestroy(&y3);CHKERRQ(ierr);
  ierr = VecDestroy(&y4);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     build:
       requires: hdf5

     test:
       nsize: 1

     test:
       nsize: 2
       suffix: 2

     test:
       nsize: 3
       suffix: 3

     test:
       nsize: 4
       suffix: 4

TEST*/
