static char help[] = "Tests VecSetValuesBlocked() on MPI vectors.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    size,rank;
  PetscInt       i,n = 8,bs = 2,indices[2];
  PetscScalar    values[4];
  Vec            x;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCheck(size == 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must be run with two processors");

  /* create vector */
  PetscCall(VecCreate(PETSC_COMM_WORLD,&x));
  PetscCall(VecSetSizes(x,PETSC_DECIDE,n));
  PetscCall(VecSetBlockSize(x,bs));
  PetscCall(VecSetFromOptions(x));

  if (rank == 0) {
    for (i=0; i<4; i++) values[i] = i+1;
    indices[0] = 0;
    indices[1] = 2;
    PetscCall(VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  /*
      Resulting vector should be 1 2 0 0 3 4 0 0
  */
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* test insertion with negative indices */
  PetscCall(VecSetOption(x,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  if (rank == 0) {
    for (i=0; i<4; i++) values[i] = -(i+1);
    indices[0] = -1;
    indices[1] = 3;
    PetscCall(VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  /*
      Resulting vector should be 1 2 0 0 3 4 -3 -4
  */
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
