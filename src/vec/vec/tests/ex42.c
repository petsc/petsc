
static char help[] = "Scatters from a parallel vector to a parallel vector.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5,N,i;
  PetscMPIInt    size,rank;
  PetscScalar    value,zero = 0.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  N    = size*n;
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&y));
  CHKERRQ(VecSetSizes(y,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(y));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,n,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(x));

  /* create two index sets */
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,n*rank,1,&is1));
  CHKERRQ(ISCreateStride(PETSC_COMM_WORLD,n,(n*(rank+1))%N,1,&is2));

  /* fill local part of parallel vector x */
  value = (PetscScalar)(rank+1);
  for (i=n*rank; i<n*(rank+1); i++) {
    CHKERRQ(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  CHKERRQ(VecSet(y,zero));

  CHKERRQ(VecScatterCreate(x,is1,y,is2,&ctx));
  for (i=0; i<100; i++) {
    PetscReal ynorm;
    PetscInt  j;
    CHKERRQ(VecNormBegin(y,NORM_2,&ynorm));
    CHKERRQ(PetscCommSplitReductionBegin(PetscObjectComm((PetscObject)y)));
    for (j=0; j<3; j++) {
      CHKERRQ(VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
      CHKERRQ(VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
    }
    CHKERRQ(VecNormEnd(y,NORM_2,&ynorm));
    /* CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"ynorm = %8.2G\n",ynorm)); */
  }
  CHKERRQ(VecScatterDestroy(&ctx));
  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));

  ierr = PetscFinalize();
  return ierr;
}
