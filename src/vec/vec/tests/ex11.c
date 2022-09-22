
static char help[] = "Scatters from a parallel vector to a sequential vector.\n\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscMPIInt size, rank;
  PetscInt    i, N;
  PetscScalar value;
  Vec         x, y;
  IS          is1, is2;
  VecScatter  ctx = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  /* create two vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, rank + 1, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecGetSize(x, &N));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, N - rank, &y));

  /* create two index sets */
  PetscCall(ISCreateStride(PETSC_COMM_SELF, N - rank, rank, 1, &is1));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, N - rank, 0, 1, &is2));

  /* fill parallel vector: note this is not efficient way*/
  for (i = 0; i < N; i++) {
    value = (PetscScalar)i;
    PetscCall(VecSetValues(x, 1, &i, &value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));
  PetscCall(VecSet(y, -1.0));

  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecScatterCreate(x, is1, y, is2, &ctx));
  PetscCall(VecScatterBegin(ctx, x, y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx, x, y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  if (rank == 0) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "----\n"));
    PetscCall(VecView(y, PETSC_VIEWER_STDOUT_SELF));
  }

  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

   test:
      suffix: bts
      nsize: 2
      args: -vec_assembly_legacy
      output_file: output/ex11_1.out

TEST*/
