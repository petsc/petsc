
static char help[] = "Tests VecSetValuesBlocked() on sequential vectors.\n\n";

#include <petscvec.h>

int main(int argc,char **argv)
{
  PetscMPIInt    size;
  PetscInt       n = 9,bs = 3,indices[2],i;
  PetscScalar    values[6];
  Vec            x;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));

  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"Must be run with one processor");

  /* create vector */
  PetscCall(VecCreate(PETSC_COMM_SELF,&x));
  PetscCall(VecSetSizes(x,n,n));
  PetscCall(VecSetBlockSize(x,bs));
  PetscCall(VecSetType(x,VECSEQ));

  for (i=0; i<6; i++) values[i] = 4.0*i;
  indices[0] = 0;
  indices[1] = 2;

  PetscCall(VecSetValuesBlocked(x,2,indices,values,INSERT_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  /*
      Resulting vector should be 0 4 8  0 0 0 12 16 20
  */
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  /* test insertion with negative indices */
  PetscCall(VecSetOption(x,VEC_IGNORE_NEGATIVE_INDICES,PETSC_TRUE));
  for (i=0; i<6; i++) values[i] = -4.0*i;
  indices[0] = -1;
  indices[1] = 2;

  PetscCall(VecSetValuesBlocked(x,2,indices,values,ADD_VALUES));
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  /*
      Resulting vector should be 0 4 8  0 0 0 0 0 0
  */
  PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&x));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
