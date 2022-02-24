
static char help[] = "Scatters from a parallel vector to a sequential vector.\n\
This does case when we are merely selecting the local part of the\n\
parallel vector.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscMPIInt    size,rank;
  PetscInt       n = 5,i;
  PetscScalar    value;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
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
  for (i=0; i<n*size; i++) {
    value = (PetscScalar) i;
    CHKERRQ(VecSetValues(x,1,&i,&value,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecScatterCreate(x,is1,y,is2,&ctx));
  CHKERRQ(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));

  if (rank == 0) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"----\n"));
    CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_SELF));
  }

  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

     test:
       nsize: 2

TEST*/
