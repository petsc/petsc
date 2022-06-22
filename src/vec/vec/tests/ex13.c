
static char help[] = "Demonstrates scattering with the indices specified by a process that is not sender or receiver.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank,size;
  Vec            x,y;
  IS             is1,is2;
  PetscInt       n,N,ix[2],iy[2];
  VecScatter     ctx;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size >= 3,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This example needs at least 3 processes");

  /* create two vectors */
  n = 2;
  N = 2*size;

  PetscCall(VecCreateMPI(PETSC_COMM_WORLD,n,N,&x));
  PetscCall(VecDuplicate(x,&y));

  /* Specify indices to send from the next process in the ring */
  ix[0] = ((rank+1)*n+0) % N;
  ix[1] = ((rank+1)*n+1) % N;
  /* And put them on the process after that in the ring */
  iy[0] = ((rank+2)*n+0) % N;
  iy[1] = ((rank+2)*n+1) % N;

  /* create two index sets */
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,n,ix,PETSC_USE_POINTER,&is1));
  PetscCall(ISCreateGeneral(PETSC_COMM_WORLD,n,iy,PETSC_USE_POINTER,&is2));

  PetscCall(VecSetValue(x,rank*n,rank*n,INSERT_VALUES));
  PetscCall(VecSetValue(x,rank*n+1,rank*n+1,INSERT_VALUES));

  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD,"----\n"));

  PetscCall(VecScatterCreate(x,is1,y,is2,&ctx));
  PetscCall(VecScatterBegin(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,x,y,INSERT_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 4

TEST*/
