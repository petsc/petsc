static char help[] = "Test PetscThreadComm Reductions.\n\n";

/*T
   Concepts: PetscThreadComm^basic example: Threaded reductions
T*/

/*
  Include "petscthreadcomm.h" so that we can use the PetscThreadComm interface.
*/
#include <petscthreadcomm.h>

PetscInt    *trstarts;

PetscErrorCode reduce_kernel(PetscInt myrank,PetscScalar *a,PetscScalar *sum)
{
  PetscScalar my_sum=*sum;
  PetscInt    i;
  PetscThreadComm tcomm;

  for(i=trstarts[myrank];i < trstarts[myrank+1];i++) { my_sum += a[i];}

  PetscCommGetThreadComm(PETSC_COMM_WORLD,&tcomm);
  PetscThreadReductionKernelBegin(myrank,tcomm,THREADCOMM_SUM,PETSC_SCALAR,&my_sum,sum);
  PetscThreadReductionKernelEnd(myrank,tcomm,THREADCOMM_SUM,PETSC_SCALAR,&my_sum,sum);
  
  return 0;
}

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       nthreads,i,Q,R,nloc;
  PetscBool      S;
  PetscScalar    *a,sum=0.0;

  PetscInitialize(&argc,&argv,(char *)0,help);

  ierr = PetscThreadCommGetNThreads(PETSC_COMM_WORLD,&nthreads);CHKERRQ(ierr);

  ierr = PetscMalloc(100*sizeof(PetscScalar),&a);CHKERRQ(ierr);
  for(i=0;i<100;i++) a[i] = 1.0;
    
  ierr = PetscMalloc((nthreads+1)*sizeof(PetscInt),&trstarts);CHKERRQ(ierr);
  trstarts[0] = 0;
  Q = 100/nthreads;
  R = 100 - Q*nthreads;
  for(i=0;i<nthreads;i++) {
    S = (PetscBool)(i < R);
    nloc = S?Q+1:Q;
    trstarts[i+1] = trstarts[i] + nloc;
  }
    
  ierr = PetscThreadCommView(PETSC_COMM_WORLD,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = PetscThreadCommRunKernel(PETSC_COMM_WORLD,(PetscThreadKernel)reduce_kernel,2,a,&sum);CHKERRQ(ierr);
  ierr = PetscThreadCommBarrier(PETSC_COMM_WORLD);CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_SELF,"Sum(x) = %f\n",sum);CHKERRQ(ierr);
  PetscFinalize();
  return 0;
}
