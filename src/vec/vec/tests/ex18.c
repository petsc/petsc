
static char help[] = "Compares BLAS dots on different machines. Input\n\
arguments are\n\
  -n <length> : local vector length\n\n";

#include <petscvec.h>

int main(int argc, char **argv)
{
  PetscInt    n = 15, i;
  PetscScalar v;
  Vec         x, y;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  if (n < 5) n = 5;

  /* create two vectors */
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &y));

  for (i = 0; i < n; i++) {
    v = ((PetscReal)i) + 1.0 / (((PetscReal)i) + .35);
    PetscCall(VecSetValues(x, 1, &i, &v, INSERT_VALUES));
    v += 1.375547826473644376;
    PetscCall(VecSetValues(y, 1, &i, &v, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(y));
  PetscCall(VecAssemblyEnd(y));

  PetscCall(VecDot(x, y, &v));
  PetscCall(PetscFPrintf(PETSC_COMM_WORLD, stdout, "Vector inner product %16.12e\n", (double)PetscRealPart(v)));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
