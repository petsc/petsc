
static char help[] = "Scatters from a parallel vector to a sequential vector.  In\n\
this case processor zero is as long as the entire parallel vector; rest are zero length.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscInt       n = 5,N,low,high,iglobal,i;
  PetscMPIInt    size,rank;
  PetscScalar    value,zero = 0.0;
  Vec            x,y;
  IS             is1,is2;
  VecScatter     ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  /* create two vectors */
  N    = size*n;
  PetscCall(VecCreate(PETSC_COMM_WORLD,&y));
  PetscCall(VecSetSizes(y,PETSC_DECIDE,N));
  PetscCall(VecSetFromOptions(y));
  if (rank == 0) {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,N,&x));
  } else {
    PetscCall(VecCreateSeq(PETSC_COMM_SELF,0,&x));
  }

  /* create two index sets */
  if (rank == 0) {
    PetscCall(ISCreateStride(PETSC_COMM_SELF,N,0,1,&is1));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,N,0,1,&is2));
  } else {
    PetscCall(ISCreateStride(PETSC_COMM_SELF,0,0,1,&is1));
    PetscCall(ISCreateStride(PETSC_COMM_SELF,0,0,1,&is2));
  }

  PetscCall(VecSet(x,zero));
  PetscCall(VecGetOwnershipRange(y,&low,&high));
  for (i=0; i<n; i++) {
    iglobal = i + low; value = (PetscScalar) (i + 10*rank);
    PetscCall(VecSetValues(y,1,&iglobal,&value,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));
  PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecScatterCreate(y,is2,x,is1,&ctx));
  PetscCall(VecScatterBegin(ctx,y,x,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx,y,x,ADD_VALUES,SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF,"----\n"));
    PetscCall(VecView(x,PETSC_VIEWER_STDOUT_SELF));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3

TEST*/
