
static char help[] = "Scatters from a sequential vector to a parallel vector.\n\
This does the tricky case.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n = 5,N;
  PetscMPIInt    size,rank;
  PetscScalar    value,zero = 0.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx = 0;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  N    = size*n;
  PetscCall(VecCreate(PETSC_COMM_WORLD,&y));
  PetscCall(VecSetSizes(y,PETSC_DECIDE,N));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF,N,&x));

  /* create two index sets */
  PetscCall(ISCreateStride(PETSC_COMM_SELF,n,0,1,&is1));
  PetscCall(ISCreateStride(PETSC_COMM_SELF,n,rank,1,&is2));

  value = rank+1;
  PetscCall(VecSet(x,value));
  PetscCall(VecSet(y,zero));

  PetscCall(VecScatterCreate(x,is1,y,is2,&ctx));
  PetscCall(VecScatterBegin(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,x,y,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
