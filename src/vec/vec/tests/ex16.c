static char help[] = "Tests VecSetValuesBlocked() on MPI vectors.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       i,n = 8,bs = 2,indices[2];
  PetscScalar    values[4];
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);

  if (size != 2) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must be run with two processors");

  /* create vector */
  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,PETSC_DECIDE,n);CHKERRQ(ierr);
  ierr = VecSetBlockSize(x,bs);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  if (rank == 0) {
    for (i=0; i<4; i++) values[i] = i+1;
    indices[0] = 0;
    indices[1] = 2;
    ierr       = VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /*
      Resulting vector should be 1 2 0 0 3 4 0 0
  */
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* test insertion with negative indices */
  ierr = VecSetOption(x,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE);CHKERRQ(ierr);
  if (rank == 0) {
    for (i=0; i<4; i++) values[i] = -(i+1);
    indices[0] = -1;
    indices[1] = 3;
    ierr       = VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  /*
      Resulting vector should be 1 2 0 0 3 4 -3 -4
  */
  ierr = VecView(x,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2

TEST*/
