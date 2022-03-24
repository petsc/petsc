
static char help[]= "Scatters from a parallel vector to a sequential vector.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n = 5,i,idx2[3] = {0,2,3},idx1[3] = {0,1,2};
  PetscMPIInt    size,rank;
  PetscScalar    value;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,size*n));
  CHKERRQ(VecSetFromOptions(x));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,n,&y));

  /* create two index sets */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,3,idx1,PETSC_COPY_VALUES,&is1));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_SELF,3,idx2,PETSC_COPY_VALUES,&is2));

  /* fill local part of parallel vector */
  for (i=n*rank; i<n*(rank+1); i++) {
    value = (PetscScalar) i;
    CHKERRQ(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecSet(y,-1.0));

  CHKERRQ(VecScatterCreate(x,is1,y,is2,&ctx));
  CHKERRQ(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));

  if (rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"scattered vector\n"));
    CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));
  }
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
