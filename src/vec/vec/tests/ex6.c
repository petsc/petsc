
static char help[] = "Demonstrates a scatter with a stride and general index set.\n\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscInt    n = 6, idx1[3] = {0, 1, 2}, loc[6] = {0, 1, 2, 3, 4, 5};
  PetscScalar two = 2.0, vals[6] = {10, 11, 12, 13, 14, 15};
  Vec         x, y;
  IS          is1, is2;
  VecScatter  ctx = 0;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));

  /* create two vector */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &x));
  PetscCall(VecDuplicate(x, &y));

  /* create two index sets */
  PetscCall(ISCreateGeneral(PETSC_COMM_SELF, 3, idx1, PETSC_COPY_VALUES, &is1));
  PetscCall(ISCreateStride(PETSC_COMM_SELF, 3, 0, 2, &is2));

  PetscCall(VecSetValues(x, 6, loc, vals, INSERT_VALUES));
  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_SELF));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "----\n"));
  PetscCall(VecSet(y, two));
  PetscCall(VecScatterCreate(x, is1, y, is2, &ctx));
  PetscCall(VecScatterBegin(ctx, x, y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterEnd(ctx, x, y, INSERT_VALUES, SCATTER_FORWARD));
  PetscCall(VecScatterDestroy(&ctx));

  PetscCall(VecView(y, PETSC_VIEWER_STDOUT_SELF));

  PetscCall(ISDestroy(&is1));
  PetscCall(ISDestroy(&is2));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
