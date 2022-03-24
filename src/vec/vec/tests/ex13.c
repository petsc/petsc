
static char help[] = "Demonstrates scattering with the indices specified by a process that is not sender or receiver.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  Vec            x,y;
  IS             is1,is2;
  PetscInt       n,N,ix[2],iy[2];
  VecScatter     ctx;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size < 3,PETSC_COMM_WORLD,PETSC_ERR_ARG_OUTOFRANGE,"This example needs at least 3 processes");

  /* create two vectors */
  n = 2;
  N = 2*size;

  CHKERRQ(VecCreateMPI(PETSC_COMM_WORLD,n,N,&x));
  CHKERRQ(VecDuplicate(x,&y));

  /* Specify indices to send from the next process in the ring */
  ix[0] = ((rank+1)*n+0) % N;
  ix[1] = ((rank+1)*n+1) % N;
  /* And put them on the process after that in the ring */
  iy[0] = ((rank+2)*n+0) % N;
  iy[1] = ((rank+2)*n+1) % N;

  /* create two index sets */
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,n,ix,PETSC_USE_POINTER,&is1));
  CHKERRQ(ISCreateGeneral(PETSC_COMM_WORLD,n,iy,PETSC_USE_POINTER,&is2));

  CHKERRQ(VecSetValue(x,rank*n,rank*n,INSERT_VALUES));
  CHKERRQ(VecSetValue(x,rank*n+1,rank*n+1,INSERT_VALUES));

  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"----\n"));

  CHKERRQ(VecScatterCreate(x,is1,y,is2,&ctx));
  CHKERRQ(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  CHKERRQ(VecScatterDestroy(&ctx));

  CHKERRQ(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(ISDestroy(&is1));
  CHKERRQ(ISDestroy(&is2));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4

TEST*/
