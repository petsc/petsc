
static char help[] = "Scatters from a parallel vector to a sequential vector.\n\
uses block index sets\n\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscInt    bs = 1, n = 5, ix0[3] = {5, 7, 9}, ix1[3] = {2, 3, 4}, i, iy0[3] = {1, 2, 4}, iy1[3] = {0, 1, 3};
  PetscMPIInt size, rank;
  PetscScalar value;
  Vec         x, y;
  IS          isx, isy;
  VecScatter  ctx = 0, newctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCheck(size == 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run with 2 processors");

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  n = bs * n;

  /* create two vectors */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, size * n));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &y));

  /* create two index sets */
  if (rank == 0) {
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, 3, ix0, PETSC_COPY_VALUES, &isx));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, 3, iy0, PETSC_COPY_VALUES, &isy));
  } else {
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, 3, ix1, PETSC_COPY_VALUES, &isx));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, 3, iy1, PETSC_COPY_VALUES, &isy));
  }

  /* fill local part of parallel vector */
  for (i = n * rank; i < n * (rank + 1); i++) {
    value = (PetscScalar)i;
    PetscCall(VecSetValues(x, 1, &i, &value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  /* fill local part of parallel vector */
  for (i = 0; i < n; i++) {
    value = -(PetscScalar)(i + 100 * rank);
    PetscCall(VecSetValues(y, 1, &i, &value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));

  PetscCall(VecScatterCreate(x, isx, y, isy, &ctx));
  PetscCall(VecScatterCopy(ctx, &newctx));
  PetscCall(VecScatterDestroy(&ctx));

  PetscCall(VecScatterBegin(newctx, y, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(newctx, y, x, INSERT_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterDestroy(&newctx));

  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(ISDestroy(&isx));
  PetscCall(ISDestroy(&isy));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
