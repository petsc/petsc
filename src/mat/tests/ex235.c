static char help[] = "Test combinations of scalings, shifts and get diagonal of MATSHELL\n\n";

#include <petscmat.h>

static PetscErrorCode myMult(Mat S, Vec x, Vec y)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(S, &A));
  PetscCall(MatMult(A, x, y));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode myGetDiagonal(Mat S, Vec d)
{
  Mat A;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(S, &A));
  PetscCall(MatGetDiagonal(A, d));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode shiftandscale(Mat A, Vec *D)
{
  Vec ll, d, rr;

  PetscFunctionBegin;
  PetscCall(MatCreateVecs(A, &ll, &rr));
  PetscCall(MatCreateVecs(A, &d, NULL));
  PetscCall(VecSetRandom(ll, NULL));
  PetscCall(VecSetRandom(rr, NULL));
  PetscCall(VecSetRandom(d, NULL));
  PetscCall(MatScale(A, 3.0));
  PetscCall(MatShift(A, -4.0));
  PetscCall(MatScale(A, 8.0));
  PetscCall(MatDiagonalSet(A, d, ADD_VALUES));
  PetscCall(MatShift(A, 9.0));
  PetscCall(MatScale(A, 8.0));
  PetscCall(VecSetRandom(ll, NULL));
  PetscCall(VecSetRandom(rr, NULL));
  PetscCall(MatDiagonalScale(A, ll, rr));
  PetscCall(MatShift(A, 2.0));
  PetscCall(MatScale(A, 11.0));
  PetscCall(VecSetRandom(d, NULL));
  PetscCall(MatDiagonalSet(A, d, ADD_VALUES));
  PetscCall(VecSetRandom(ll, NULL));
  PetscCall(VecSetRandom(rr, NULL));
  PetscCall(MatDiagonalScale(A, ll, rr));
  PetscCall(MatShift(A, 5.0));
  PetscCall(MatScale(A, 7.0));
  PetscCall(MatGetDiagonal(A, d));
  *D = d;
  PetscCall(VecDestroy(&ll));
  PetscCall(VecDestroy(&rr));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat       A, Aij, B;
  Vec       Adiag, Aijdiag;
  PetscInt  m = 3;
  PetscReal Aijnorm, Aijdiagnorm, Bnorm, dnorm;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, m, 7, NULL, 6, NULL, &Aij));
  PetscCall(MatSetRandom(Aij, NULL));
  PetscCall(MatSetOption(Aij, MAT_NEW_NONZERO_LOCATION_ERR, PETSC_FALSE));

  PetscCall(MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, m, m, Aij, &A));
  PetscCall(MatShellSetOperation(A, MATOP_MULT, (void (*)(void))myMult));
  PetscCall(MatShellSetOperation(A, MATOP_GET_DIAGONAL, (void (*)(void))myGetDiagonal));

  PetscCall(shiftandscale(A, &Adiag));
  PetscCall(MatComputeOperator(A, NULL, &B));
  PetscCall(shiftandscale(Aij, &Aijdiag));
  PetscCall(MatAXPY(Aij, -1.0, B, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatNorm(Aij, NORM_FROBENIUS, &Aijnorm));
  PetscCall(MatNorm(B, NORM_FROBENIUS, &Bnorm));
  PetscCheck(Aijnorm / Bnorm <= 100.0 * PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Altered matrices do not match, norm of difference %g", (double)(Aijnorm / Bnorm));
  PetscCall(VecAXPY(Aijdiag, -1.0, Adiag));
  PetscCall(VecNorm(Adiag, NORM_2, &dnorm));
  PetscCall(VecNorm(Aijdiag, NORM_2, &Aijdiagnorm));
  PetscCheck(Aijdiagnorm / dnorm <= 100.0 * PETSC_MACHINE_EPSILON, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Altered matrices diagonals do not match, norm of difference %g", (double)(Aijdiagnorm / dnorm));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&Aij));
  PetscCall(VecDestroy(&Adiag));
  PetscCall(VecDestroy(&Aijdiag));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      nsize: {{1 2 3 4}}

TEST*/
