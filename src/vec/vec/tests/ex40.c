
static char help[] = "Tests taking part of existing array to create a new vector.\n\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscMPIInt size;
  PetscInt    n = 10, i;
  PetscScalar array[10];
  Vec         x;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* create vector */
  for (i = 0; i < n; i++) array[i] = i;
  n = n - 1;

  PetscCall(VecCreateSeqWithArray(PETSC_COMM_SELF, 1, n, array + 1, &x));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(VecDestroy(&x));
  PetscCall(PetscFinalize());
  return 0;
}
