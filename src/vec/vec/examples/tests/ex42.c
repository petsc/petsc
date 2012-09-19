
static char help[] = "Scatters from a parallel vector to a parallel vector.\n\n";

#include <petscvec.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5,N,i;
  PetscMPIInt    size,rank;
  PetscScalar    value,zero = 0.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRQ(ierr);

  /* create two vectors */
  N = size*n;
  ierr = VecCreate(PETSC_COMM_WORLD,&y);CHKERRQ(ierr);
  ierr = VecSetSizes(y,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(y);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
  ierr = VecSetSizes(x,n,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);CHKERRQ(ierr);

  /* create two index sets */
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&is1);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_WORLD,n,(n*(rank+1))%N,1,&is2);CHKERRQ(ierr);

  /* fill local part of parallel vector x */
  value = (PetscScalar)(rank+1);
  for (i=n*rank; i<n*(rank+1); i++) {
    ierr = VecSetValues(x,1,&i,&value,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = VecAssemblyBegin(x);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(x);CHKERRQ(ierr);

  ierr = VecSet(y,zero);CHKERRQ(ierr);

  ierr = VecScatterCreate(x,is1,y,is2,&ctx);CHKERRQ(ierr);
  for (i=0; i<100; i++) {
    PetscReal ynorm;
    PetscInt j;
    ierr = VecNormBegin(y,NORM_2,&ynorm);CHKERRQ(ierr);
    ierr = PetscCommSplitReductionBegin(((PetscObject)y)->comm);CHKERRQ(ierr);
    for (j=0; j<3; j++) {
      ierr = VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
      ierr = VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_FORWARD);CHKERRQ(ierr);
    }
    ierr = VecNormEnd(y,NORM_2,&ynorm);CHKERRQ(ierr);
    /* ierr = PetscPrintf(PETSC_COMM_WORLD,"ynorm = %8.2G\n",ynorm);CHKERRQ(ierr); */
  }
  ierr = VecScatterDestroy(&ctx);CHKERRQ(ierr);
  ierr = VecView(y,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = ISDestroy(&is1);CHKERRQ(ierr);
  ierr = ISDestroy(&is2);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

