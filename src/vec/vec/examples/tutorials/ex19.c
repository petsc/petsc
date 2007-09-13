static char help[] = "Parallel HDF5 Vec Viewing.\n\n";

/*T
   Concepts: vectors^viewing
   Concepts: viewers^hdf5
   Processors: n
T*/

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  Vec            x;
  PetscViewer    viewer;
  PetscMPIInt    rank;
  PetscInt       i, nlocal, n = 6;
  PetscScalar   *array;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscInitialize(&argc, &argv, (char *) 0, help);CHKERRQ(ierr); 
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD, &rank);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL, "-n", &n, PETSC_NULL);CHKERRQ(ierr);
  
  ierr = VecCreate(PETSC_COMM_WORLD, &x);CHKERRQ(ierr);
  ierr = VecSetSizes(x, PETSC_DECIDE, n);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  ierr = VecGetLocalSize(x, &nlocal);CHKERRQ(ierr);
  ierr = VecGetArray(x, &array);CHKERRQ(ierr);
  for(i = 0; i < nlocal; i++) {
    array[i] = rank + 1;
  }
  ierr = VecRestoreArray(x, &array);CHKERRQ(ierr);
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, "ex19.h5", FILE_MODE_WRITE, &viewer);CHKERRQ(ierr);
  ierr = VecView(x, viewer);CHKERRQ(ierr);

  ierr = PetscViewerDestroy(viewer);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
 
