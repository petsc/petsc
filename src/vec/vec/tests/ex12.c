
static char help[] = "Scatters from a sequential vector to a parallel vector.\n\
This does case when we are merely selecting the local part of the\n\
parallel vector.\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    size,rank;
  PetscInt       n = 5,i;
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
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,n*rank,1,&is1));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,0,1,&is2));

  /* each processor inserts the entire vector */
  /* this is redundant but tests assembly */
  for (i=0; i<n; i++) {
    value = (PetscScalar) (i + 10*rank);
    CHKERRQ(VecSetValues(y,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(y));
  CHKERRQ(VecAssemblyEnd(y));

  CHKERRQ(VecScatterCreate(y,is2,x,is1,&ctx));
  CHKERRQ(VecScatterBegin(ctx,y,x,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,y,x,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
