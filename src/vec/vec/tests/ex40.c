
static char help[] = "Tests taking part of existing array to create a new vector.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size;
  PetscInt       n = 10,i;
  PetscScalar    array[10];
  Vec            x;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
  PetscAssertFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");

  /* create vector */
  for (i=0; i<n; i++) array[i] = i;
  n = n-1;

  ierr = VecCreateSeqWithArray(PETSC_COMM_SELF,1,n,array+1,&x);CHKERRQ(ierr);
  ierr = VecView(x,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

