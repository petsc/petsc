static char help[] = "Test to demonstrate interface for thread reductions and passing scalar values.\n\n";

/*T
   Concepts: PetscThreadComm^basic example: Threaded reductions and passing scalar values
T*/

/*
  Include "petscthreadcomm.h" so that we can use the PetscThreadComm interface.
*/
#include <petscthreadcomm.h>

PetscInt    *trstarts;

PetscErrorCode set_kernel(PetscInt myrank,PetscScalar *a,PetscScalar *alphap)
{
  PetscScalar alpha=*alphap;
  PetscInt    i;

  for(i=trstarts[myrank];i < trstarts[myrank+1];i++) a[i] = alpha;

  return 0;
}

PetscErrorCode reduce_kernel(PetscInt myrank,PetscScalar *a,PetscThreadCommRedCtx red)
{
  PetscScalar my_sum=0.0;
  PetscInt    i;

  for(i=trstarts[myrank];i < trstarts[myrank+1];i++) my_sum += a[i];

  PetscThreadReductionKernelBegin(myrank,red,&my_sum);
  
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode         ierr;
  PetscInt               N=64;
  PetscScalar           *a,sum=0.0,alpha=1.0,*scalar;
  PetscThreadCommRedCtx  red;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = PetscThreadCommView(PETSC_COMM_WORLD,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscMalloc(N*sizeof(PetscScalar),&a);CHKERRQ(ierr);
    
  /* Set thread ownership ranges for the array */
  ierr = PetscThreadCommGetOwnershipRanges(PETSC_COMM_WORLD,N,&trstarts);CHKERRQ(ierr);

  /* Set a[i] = 1.0 .. i = 1,N */
  /* Get location to store the scalar value alpha from threadcomm */
  ierr = PetscThreadCommGetScalars(PETSC_COMM_WORLD,&scalar,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  *scalar = alpha;
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)set_kernel,2,a,scalar);CHKERRQ(ierr);

  ierr = PetscThreadReductionBegin(PETSC_COMM_WORLD,THREADCOMM_SUM,PETSC_SCALAR,&red);CHKERRQ(ierr);
  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)reduce_kernel,2,a,red);CHKERRQ(ierr);
  ierr = PetscThreadReductionEnd(red,&sum);CHKERRQ(ierr);

  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Sum(x) = %f\n",sum);CHKERRQ(ierr);
  ierr = PetscFree(a);CHKERRQ(ierr);
  ierr = PetscFree(trstarts);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
