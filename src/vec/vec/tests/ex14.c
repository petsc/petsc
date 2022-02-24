
static char help[] = "Scatters from a sequential vector to a parallel vector.\n\
This does the tricky case.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       n = 5,N;
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
  CHKERRQ(VecSetSizes(y,PETSC_DECIDE,N));
  CHKERRQ(VecSetFromOptions(y));
  CHKERRQ(VecCreateSeq(PETSC_COMM_SELF,N,&x));

  /* create two index sets */
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,0,1,&is1));
  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,n,rank,1,&is2));

  value = rank+1;
  CHKERRQ(VecSet(x,value));
  CHKERRQ(VecSet(y,zero));

  CHKERRQ(VecScatterCreate(x,is1,y,is2,&ctx));
  CHKERRQ(VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));

  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

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
