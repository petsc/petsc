static char help[] = "Test different MatSolve routines with MATTRANSPOSEVIRTUAL.\n\n";

#include <petscmat.h>

PetscErrorCode TestMatrix(const char *test, Mat A, PetscInt nrhs, PetscBool inplace, PetscBool chol)
{
  Mat       F, RHS, X, C1;
  Vec       b, x, y, f;
  IS        perm, iperm;
  PetscInt  n, i;
  PetscReal norm, tol = 1000 * PETSC_MACHINE_EPSILON;
  PetscBool ht;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar v1 = PetscCMPLX(1.0, -0.1), v2 = PetscCMPLX(-1.0, 0.1);
#else
  PetscScalar v1 = 1.0, v2 = -1.0;
#endif

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)A, MATHERMITIANTRANSPOSEVIRTUAL, &ht));
  PetscCall(MatCreateVecs(A, &f, &b));
  PetscCall(MatCreateVecs(A, &x, &y));
  PetscCall(VecSet(b, v1));
  PetscCall(VecSet(y, v2));

  PetscCall(MatGetOrdering(A, MATORDERINGND, &perm, &iperm));
  if (!inplace) {
    if (!chol) {
      PetscCall(MatGetFactor(A, MATSOLVERPETSC, MAT_FACTOR_LU, &F));
      PetscCall(MatLUFactorSymbolic(F, A, perm, iperm, NULL));
      PetscCall(MatLUFactorNumeric(F, A, NULL));
    } else { /* Cholesky */
      PetscCall(MatGetFactor(A, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &F));
      PetscCall(MatCholeskyFactorSymbolic(F, A, perm, NULL));
      PetscCall(MatCholeskyFactorNumeric(F, A, NULL));
    }
  } else { /* Test inplace factorization */
    PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &F));
    if (!chol) PetscCall(MatLUFactor(F, perm, iperm, NULL));
    else PetscCall(MatCholeskyFactor(F, perm, NULL));
  }

  /* MatSolve */
  PetscCall(MatSolve(F, b, x));
  PetscCall(MatMult(A, x, f));
  PetscCall(VecAXPY(f, -1.0, b));
  PetscCall(VecNorm(f, NORM_2, &norm));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12s MatSolve              : Error of norm %g\n", test, (double)norm));

  /* MatSolveTranspose */
  if (!ht) {
    PetscCall(MatSolveTranspose(F, b, x));
    PetscCall(MatMultTranspose(A, x, f));
    PetscCall(VecAXPY(f, -1.0, b));
    PetscCall(VecNorm(f, NORM_2, &norm));
    if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12s MatSolveTranspose     : Error of norm %g\n", test, (double)norm));
  }

  /* MatSolveAdd */
  PetscCall(MatSolveAdd(F, b, y, x));
  PetscCall(MatMult(A, y, f));
  PetscCall(VecScale(f, -1.0));
  PetscCall(MatMultAdd(A, x, f, f));
  PetscCall(VecAXPY(f, -1.0, b));
  PetscCall(VecNorm(f, NORM_2, &norm));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12s MatSolveAdd           : Error of norm %g\n", test, (double)norm));

  /* MatSolveTransposeAdd */
  if (!ht) {
    PetscCall(MatSolveTransposeAdd(F, b, y, x));
    PetscCall(MatMultTranspose(A, y, f));
    PetscCall(VecScale(f, -1.0));
    PetscCall(MatMultTransposeAdd(A, x, f, f));
    PetscCall(VecAXPY(f, -1.0, b));
    PetscCall(VecNorm(f, NORM_2, &norm));
    if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12s MatSolveTransposeAdd  : Error of norm %g\n", test, (double)norm));
  }

  /* MatMatSolve */
  PetscCall(MatGetSize(A, &n, NULL));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &RHS));
  PetscCall(MatSetSizes(RHS, PETSC_DECIDE, PETSC_DECIDE, n, nrhs));
  PetscCall(MatSetType(RHS, MATSEQDENSE));
  PetscCall(MatSetUp(RHS));
  for (i = 0; i < nrhs; i++) PetscCall(MatSetValue(RHS, i, i, 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(RHS, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(RHS, MAT_FINAL_ASSEMBLY));
  PetscCall(MatDuplicate(RHS, MAT_DO_NOT_COPY_VALUES, &X));

  if (!ht) {
    PetscCall(MatMatSolve(F, RHS, X));
    PetscCall(MatMatMult(A, X, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C1));
    PetscCall(MatAXPY(C1, -1.0, RHS, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(C1, NORM_FROBENIUS, &norm));
    if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12s MatMatSolve           : Error of norm %g\n", test, (double)norm));
    PetscCall(MatDestroy(&C1));

    PetscCall(MatMatSolveTranspose(F, RHS, X));
    PetscCall(MatTransposeMatMult(A, X, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &C1));
    PetscCall(MatAXPY(C1, -1.0, RHS, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(C1, NORM_FROBENIUS, &norm));
    if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "%12s MatMatSolveTranspose  : Error of norm %g\n", test, (double)norm));
    PetscCall(MatDestroy(&C1));
  }
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&f));
  PetscCall(VecDestroy(&y));
  PetscCall(ISDestroy(&perm));
  PetscCall(ISDestroy(&iperm));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&RHS));
  PetscCall(MatDestroy(&X));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  PetscMPIInt size;
  Mat         A, At, Aht;
  PetscInt    i, n = 8, nrhs = 2;
  PetscBool   aij, inplace = PETSC_FALSE;
#if defined(PETSC_USE_COMPLEX)
  PetscScalar a = PetscCMPLX(-1.0, 0.5);
#else
  PetscScalar a = -1.0;
#endif

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-inplace", &inplace, NULL));
  PetscCheck(nrhs <= n, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "Must have nrhs <= n");

  /* Hermitian matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  for (i = 0; i < n; i++) {
    if (i > 0) PetscCall(MatSetValue(A, i, i - 1, a, INSERT_VALUES));
    if (i < n - 1) PetscCall(MatSetValue(A, i, i + 1, PetscConj(a), INSERT_VALUES));
    PetscCall(MatSetValue(A, i, i, 2.0, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(PetscObjectTypeCompareAny((PetscObject)A, &aij, MATSEQAIJ, MATSEQBAIJ, ""));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatSetOption(A, MAT_HERMITIAN, PETSC_TRUE));
#else
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_TRUE));
#endif

  PetscCall(MatCreateTranspose(A, &At));
  PetscCall(MatCreateHermitianTranspose(A, &Aht));

  PetscCall(TestMatrix("LU T", At, nrhs, inplace, PETSC_FALSE));
  PetscCall(TestMatrix("LU HT", Aht, nrhs, inplace, PETSC_FALSE));
  if (!aij) {
    PetscCall(TestMatrix("Chol T", At, nrhs, inplace, PETSC_TRUE));
    PetscCall(TestMatrix("Chol HT", Aht, nrhs, inplace, PETSC_TRUE));
  }

  /* Make the matrix non-Hermitian */
  PetscCall(MatSetValue(A, 0, 1, -5.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
#if defined(PETSC_USE_COMPLEX)
  PetscCall(MatSetOption(A, MAT_HERMITIAN, PETSC_FALSE));
#else
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, PETSC_FALSE));
#endif

  PetscCall(TestMatrix("LU T nonsym", At, nrhs, inplace, PETSC_FALSE));
  PetscCall(TestMatrix("LU HT nonsym", Aht, nrhs, inplace, PETSC_FALSE));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&At));
  PetscCall(MatDestroy(&Aht));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: 1
      args: -inplace {{0 1}} -mat_type {{aij dense}}
      output_file: output/empty.out

TEST*/
