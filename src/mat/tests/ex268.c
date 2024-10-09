static char help[] = "Tests MATFACTORHTOOL\n\n";

#include <petscmat.h>

static PetscErrorCode GenEntries(PetscInt sdim, PetscInt M, PetscInt N, const PetscInt *J, const PetscInt *K, PetscScalar *ptr, void *ctx)
{
  PetscInt  d, j, k;
  PetscReal diff = 0.0, *coords = (PetscReal *)(ctx);

  PetscFunctionBeginUser;
  for (j = 0; j < M; j++) {
    for (k = 0; k < N; k++) {
      diff = 0.0;
      for (d = 0; d < sdim; d++) diff += (coords[J[j] * sdim + d] - coords[K[k] * sdim + d]) * (coords[J[j] * sdim + d] - coords[K[k] * sdim + d]);
      ptr[j + M * k] = 1.0 / (1.0e-1 + PetscSqrtReal(diff));
    }
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat               A, Ad, F, Fd, X, Xd, B;
  Vec               x, xd, b;
  PetscInt          m = 100, dim = 3, M, K = 10, begin, n = 0;
  PetscMPIInt       size;
  PetscReal        *coords, *gcoords, norm, epsilon;
  MatHtoolKernelFn *kernel = GenEntries;
  PetscBool         flg, sym = PETSC_FALSE;
  PetscRandom       rdm;
  MatSolverType     type;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m_local", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_local", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-dim", &dim, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-K", &K, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-symmetric", &sym, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-mat_htool_epsilon", &epsilon, NULL));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  M = size * m;
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscMalloc1(m * dim, &coords));
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rdm));
  PetscCall(PetscRandomGetValuesReal(rdm, m * dim, coords));
  PetscCall(PetscCalloc1(M * dim, &gcoords));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, M, K, NULL, &B));
  PetscCall(MatSetRandom(B, rdm));
  PetscCall(MatGetOwnershipRange(B, &begin, NULL));
  PetscCall(PetscArraycpy(gcoords + begin * dim, coords, m * dim));
  PetscCallMPI(MPIU_Allreduce(MPI_IN_PLACE, gcoords, M * dim, MPIU_REAL, MPI_SUM, PETSC_COMM_WORLD));
  PetscCall(MatCreateHtoolFromKernel(PETSC_COMM_WORLD, m, m, M, M, dim, coords, coords, kernel, gcoords, &A));
  PetscCall(MatSetOption(A, MAT_SYMMETRIC, sym));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatConvert(A, MATDENSE, MAT_INITIAL_MATRIX, &Ad));
  PetscCall(MatMultEqual(A, Ad, 10, &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Ax != Adx");
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, M, K, NULL, &X));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, M, K, NULL, &Xd));
  PetscCall(MatViewFromOptions(A, NULL, "-A"));
  PetscCall(MatViewFromOptions(Ad, NULL, "-Ad"));
  PetscCall(MatViewFromOptions(B, NULL, "-B"));
  for (PetscInt i = 0; i < 2; ++i) {
    PetscCall(MatGetFactor(A, MATSOLVERHTOOL, i == 0 ? MAT_FACTOR_LU : MAT_FACTOR_CHOLESKY, &F));
    PetscCall(MatGetFactor(Ad, MATSOLVERPETSC, i == 0 ? MAT_FACTOR_LU : MAT_FACTOR_CHOLESKY, &Fd));
    PetscCall(MatFactorGetSolverType(F, &type));
    PetscCall(PetscStrncmp(type, MATSOLVERHTOOL, 5, &flg));
    PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "MATSOLVERHTOOL != htool");
    if (i == 0) {
      PetscCall(MatLUFactorSymbolic(F, A, NULL, NULL, NULL));
      PetscCall(MatLUFactorNumeric(F, A, NULL));
      PetscCall(MatLUFactorSymbolic(Fd, Ad, NULL, NULL, NULL));
      PetscCall(MatLUFactorNumeric(Fd, Ad, NULL));
    } else {
      PetscCall(MatCholeskyFactorSymbolic(F, A, NULL, NULL));
      PetscCall(MatCholeskyFactorNumeric(F, A, NULL));
      PetscCall(MatCholeskyFactorSymbolic(Fd, Ad, NULL, NULL));
      PetscCall(MatCholeskyFactorNumeric(Fd, Ad, NULL));
    }
    PetscCall(MatMatSolve(F, B, X));
    PetscCall(MatMatSolve(Fd, B, Xd));
    PetscCall(MatViewFromOptions(X, NULL, "-X"));
    PetscCall(MatViewFromOptions(Xd, NULL, "-Xd"));
    PetscCall(MatAXPY(Xd, -1.0, X, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(Xd, NORM_INFINITY, &norm));
    PetscCall(MatViewFromOptions(Xd, NULL, "-MatMatSolve"));
    if (norm > 0.01) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: norm of residual for MatMatSolve %g\n", (double)norm));
    if (!PetscDefined(USE_COMPLEX) || i == 0) {
      PetscCall(MatMatSolveTranspose(F, B, X));
      PetscCall(MatMatSolveTranspose(Fd, B, Xd));
      PetscCall(MatAXPY(Xd, -1.0, X, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(Xd, NORM_INFINITY, &norm));
      PetscCall(MatViewFromOptions(Xd, NULL, "-MatMatSolveTranspose"));
      if (norm > 0.01) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: norm of residual for MatMatSolveTranspose %g\n", (double)norm));
    }
    PetscCall(MatDenseGetColumnVecRead(B, 0, &b));
    PetscCall(MatDenseGetColumnVecWrite(X, 0, &x));
    PetscCall(MatDenseGetColumnVecWrite(Xd, 0, &xd));
    PetscCall(MatSolve(F, b, x));
    PetscCall(MatSolve(Fd, b, xd));
    PetscCall(VecAXPY(xd, -1.0, x));
    PetscCall(VecNorm(xd, NORM_INFINITY, &norm));
    PetscCall(MatViewFromOptions(Xd, NULL, "-MatSolve"));
    if (norm > 0.01) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: norm of residual for MatSolve %g\n", (double)norm));
    if (!PetscDefined(USE_COMPLEX) || i == 0) {
      PetscCall(MatSolveTranspose(F, b, x));
      PetscCall(MatSolveTranspose(Fd, b, xd));
      PetscCall(VecAXPY(xd, -1.0, x));
      PetscCall(VecNorm(xd, NORM_INFINITY, &norm));
      PetscCall(MatViewFromOptions(Xd, NULL, "-MatSolveTranspose"));
      if (norm > 0.01) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: norm of residual for MatSolveTranspose %g\n", (double)norm));
    }
    PetscCall(MatDenseRestoreColumnVecWrite(Xd, 0, &xd));
    PetscCall(MatDenseRestoreColumnVecWrite(X, 0, &x));
    PetscCall(MatDenseRestoreColumnVecRead(B, 0, &b));
    PetscCall(MatDestroy(&Fd));
    PetscCall(MatDestroy(&F));
  }
  PetscCall(MatDestroy(&Xd));
  PetscCall(MatDestroy(&X));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(MatDestroy(&Ad));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFree(gcoords));
  PetscCall(PetscFree(coords));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   build:
      requires: htool

   test:
      requires: htool
      suffix: 1
      nsize: 1
      args: -mat_htool_epsilon 1.0e-11
      output_file: output/empty.out

TEST*/
