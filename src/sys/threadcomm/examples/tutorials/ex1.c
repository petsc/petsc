static char help[] = "Test PetscThreadComm Interface.\n\n";

/*T
   Concepts: PetscThreadComm^basic example
T*/

/*
  Include "petscthreadcomm.h" so that we can use the PetscThreadComm interface.
*/
#include <petscthreadcomm.h>

PetscErrorCode kernel_func1(PetscInt myrank,PetscInt *ranks,PetscScalar *values)
{
  values[myrank] *= 1;
  printf("First Kernel:My rank is %d, x = %f\n",ranks[myrank],values[myrank]); 
  return(0);
}

PetscErrorCode kernel_func2(PetscInt myrank,PetscInt *ranks,PetscScalar *values)
{
  values[myrank] *= 2;
  printf("Second Kernel:My rank is %d, x = %f\n",ranks[myrank],values[myrank]); 
  return(0);

}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       nthreads,i;
  PetscInt       *ranks;
  PetscScalar    *values;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = PetscThreadCommView(PETSC_COMM_WORLD,0);CHKERRQ(ierr);
  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);

  ierr = PetscMalloc(nthreads*sizeof(PetscInt),&ranks);CHKERRQ(ierr);
  ierr = PetscMalloc(nthreads*sizeof(PetscScalar),&values);CHKERRQ(ierr);

  for(i=0;i < nthreads;i++) {
    ranks[i] = i; values[i] = i;
  }

  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)kernel_func1,2,ranks,values);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)kernel_func2,2,ranks,values);CHKERRQ(ierr);

  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);
  ierr = PetscFree(ranks);CHKERRQ(ierr);
  ierr = PetscFree(values);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
