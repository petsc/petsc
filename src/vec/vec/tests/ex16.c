static char help[] = "Tests VecSetValuesBlocked() on MPI vectors.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    size,rank;
  PetscInt       i,n = 8,bs = 2,indices[2];
  PetscScalar    values[4];
  Vec            x;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));

  PetscCheckFalse(size != 2,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must be run with two processors");

  /* create vector */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&x));
  CHKERRQ(VecSetSizes(x,PETSC_DECIDE,n));
  CHKERRQ(VecSetBlockSize(x,bs));
  CHKERRQ(VecSetFromOptions(x));

  if (rank == 0) {
    for (i=0; i<4; i++) values[i] = i+1;
    indices[0] = 0;
    indices[1] = 2;
    CHKERRQ(VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /*
      Resulting vector should be 1 2 0 0 3 4 0 0
  */
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* test insertion with negative indices */
  CHKERRQ(VecSetOption(x,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  if (rank == 0) {
    for (i=0; i<4; i++) values[i] = -(i+1);
    indices[0] = -1;
    indices[1] = 3;
    CHKERRQ(VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES));
  }
  CHKERRQ(VecAssemblyBegin(x));
  CHKERRQ(VecAssemblyEnd(x));

  /*
      Resulting vector should be 1 2 0 0 3 4 -3 -4
  */
  CHKERRQ(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&x));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
