
static char help[] = "Scatters between parallel vectors. \n\
uses block index sets\n\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscInt       bs = 1, n = 5, N, i, low;
  PetscInt       ix0[3] = {5, 7, 9}, iy0[3] = {1, 2, 4}, ix1[3] = {2, 3, 1}, iy1[3] = {0, 3, 9};
  PetscMPIInt    size, rank;
  PetscScalar   *array;
  Vec            x, x1, y;
  IS             isx, isy;
  VecScatter     ctx;
  VecScatterType type;
  PetscBool      flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));

  PetscCheck(size >= 2, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "Must run more than one processor");

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  n = bs * n;

  /* Create vector x over shared memory */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(x));

  PetscCall(VecGetOwnershipRange(x, &low, NULL));
  PetscCall(VecGetArray(x, &array));
  for (i = 0; i < n; i++) array[i] = (PetscScalar)(i + low);
  PetscCall(VecRestoreArray(x, &array));
  /* PetscCall(VecView(x,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Test some vector functions */
  PetscCall(VecAssemblyBegin(x));
  PetscCall(VecAssemblyEnd(x));

  PetscCall(VecGetSize(x, &N));
  PetscCall(VecGetLocalSize(x, &n));

  PetscCall(VecDuplicate(x, &x1));
  PetscCall(VecCopy(x, x1));
  PetscCall(VecEqual(x, x1, &flg));
  PetscCheck(flg, PetscObjectComm((PetscObject)x), PETSC_ERR_ARG_WRONG, "x1 != x");

  PetscCall(VecScale(x1, 2.0));
  PetscCall(VecSet(x1, 10.0));
  /* PetscCall(VecView(x1,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Create vector y over shared memory */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &y));
  PetscCall(VecSetSizes(y, n, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(y));
  PetscCall(VecGetArray(y, &array));
  for (i = 0; i < n; i++) array[i] = -(PetscScalar)(i + 100 * rank);
  PetscCall(VecRestoreArray(y, &array));
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));
  /* PetscCall(VecView(y,PETSC_VIEWER_STDOUT_WORLD)); */

  /* Create two index sets */
  if (rank == 0) {
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, 3, ix0, PETSC_COPY_VALUES, &isx));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, 3, iy0, PETSC_COPY_VALUES, &isy));
  } else {
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, 3, ix1, PETSC_COPY_VALUES, &isx));
    PetscCall(ISCreateBlock(PETSC_COMM_SELF, bs, 3, iy1, PETSC_COPY_VALUES, &isy));
  }

  if (rank == 10) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n[%d] isx:\n", rank));
    PetscCall(ISView(isx, PETSC_VIEWER_STDOUT_SELF));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n[%d] isy:\n", rank));
    PetscCall(ISView(isy, PETSC_VIEWER_STDOUT_SELF));
  }

  /* Create Vector scatter */
  PetscCall(VecScatterCreate(x, isx, y, isy, &ctx));
  PetscCall(VecScatterSetFromOptions(ctx));
  PetscCall(VecScatterGetType(ctx, &type));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "scatter type %s\n", type));

  /* Test forward vecscatter */
  PetscCall(VecSet(y, 0.0));
  PetscCall(VecScatterBegin(ctx, x, y, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx, x, y, ADD_VALUES, SCATTER_FORWARD));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nSCATTER_FORWARD y:\n"));
  PetscCall(VecView(y, PETSC_VIEWER_STDOUT_WORLD));

  /* Test reverse vecscatter */
  PetscCall(VecSet(x, 0.0));
  PetscCall(VecSet(y, 0.0));
  PetscCall(VecGetOwnershipRange(y, &low, NULL));
  PetscCall(VecGetArray(y, &array));
  for (i = 0; i < n; i++) array[i] = (PetscScalar)(i + low);
  PetscCall(VecRestoreArray(y, &array));
  PetscCall(VecScatterBegin(ctx, y, x, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(VecScatterEnd(ctx, y, x, ADD_VALUES, SCATTER_REVERSE));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "\nSCATTER_REVERSE x:\n"));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  /* Free objects */
  PetscCall(VecScatterDestroy(&ctx));
  PetscCall(ISDestroy(&isx));
  PetscCall(ISDestroy(&isy));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x1));
  PetscCall(VecDestroy(&y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 2

TEST*/
