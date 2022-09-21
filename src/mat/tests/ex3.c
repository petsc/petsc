
static char help[] = "Tests relaxation for dense matrices.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat         C;
  Vec         u, x, b, e;
  PetscInt    i, n = 10, midx[3];
  PetscScalar v[3];
  PetscReal   omega = 1.0, norm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-omega", &omega, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));

  PetscCall(MatCreate(PETSC_COMM_SELF, &C));
  PetscCall(MatSetSizes(C, n, n, n, n));
  PetscCall(MatSetType(C, MATSEQDENSE));
  PetscCall(MatSetUp(C));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &b));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &x));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &u));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &e));
  PetscCall(VecSet(u, 1.0));
  PetscCall(VecSet(x, 0.0));

  v[0] = -1.;
  v[1] = 2.;
  v[2] = -1.;
  for (i = 1; i < n - 1; i++) {
    midx[0] = i - 1;
    midx[1] = i;
    midx[2] = i + 1;
    PetscCall(MatSetValues(C, 1, &i, 3, midx, v, INSERT_VALUES));
  }
  i       = 0;
  midx[0] = 0;
  midx[1] = 1;
  v[0]    = 2.0;
  v[1]    = -1.;
  PetscCall(MatSetValues(C, 1, &i, 2, midx, v, INSERT_VALUES));
  i       = n - 1;
  midx[0] = n - 2;
  midx[1] = n - 1;
  v[0]    = -1.0;
  v[1]    = 2.;
  PetscCall(MatSetValues(C, 1, &i, 2, midx, v, INSERT_VALUES));

  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  PetscCall(MatMult(C, u, b));

  for (i = 0; i < n; i++) {
    PetscCall(MatSOR(C, b, omega, SOR_FORWARD_SWEEP, 0.0, 1, 1, x));
    PetscCall(VecWAXPY(e, -1.0, x, u));
    PetscCall(VecNorm(e, NORM_2, &norm));
    PetscCall(PetscPrintf(PETSC_COMM_SELF, "2-norm of error %g\n", (double)norm));
  }
  PetscCall(MatDestroy(&C));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&e));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
