static char help[] = "Tests MatMatSolve() in Schur complement mode.\n\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat           F, A, B, X, Y, S;
  IS            is_schur;
  PetscMPIInt   size;
  PetscInt      ns = 0, m, n;
  PetscReal     norm, tol = PETSC_SQRT_MACHINE_EPSILON;
  MatFactorType factor = MAT_FACTOR_LU;
  PetscViewer   fd;
  char          solver[256], converttype[256];
  char          file[PETSC_MAX_PATH_LEN];
  PetscBool     flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor test");

  PetscCall(PetscOptionsGetString(NULL, NULL, "-A", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_SUP, "Must provide a binary matrix with -A filename option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatLoad(A, fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatGetSize(A, &m, &n));
  PetscCheck(m == n, PETSC_COMM_WORLD, PETSC_ERR_ARG_SIZ, "This example is not intended for rectangular matrices (%" PetscInt_FMT ", %" PetscInt_FMT ")", m, n);
  PetscCall(MatViewFromOptions(A, NULL, "-A_view"));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-B", file, sizeof(file), &flg));
  PetscCheck(flg, PETSC_COMM_WORLD, PETSC_ERR_SUP, "Must provide a binary matrix with -B filename option");
  PetscCall(PetscViewerBinaryOpen(PETSC_COMM_WORLD, file, FILE_MODE_READ, &fd));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &B));
  PetscCall(MatLoad(B, fd));
  PetscCall(PetscViewerDestroy(&fd));
  PetscCall(MatViewFromOptions(B, NULL, "-B_view"));
  PetscCall(PetscObjectBaseTypeCompareAny((PetscObject)B, &flg, MATSEQDENSE, MATMPIDENSE, NULL));
  if (!flg) PetscCall(PetscObjectTypeCompare((PetscObject)B, MATTRANSPOSEVIRTUAL, &flg));
  if (!flg) {
    Mat Bt;

    PetscCall(MatCreateTranspose(B, &Bt));
    PetscCall(MatDestroy(&B));
    B = Bt;
  }
  PetscCall(PetscOptionsGetString(NULL, NULL, "-B_convert_type", converttype, sizeof(converttype), &flg));
  if (flg) PetscCall(MatConvert(B, converttype, MAT_INPLACE_MATRIX, &B));

  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ns", &ns, NULL));

  PetscCall(PetscOptionsGetString(NULL, NULL, "-mat_solver_type", solver, sizeof(solver), &flg));
  if (!flg) PetscCall(PetscStrncpy(solver, MATSOLVERMUMPS, sizeof(solver)));
  PetscCall(PetscOptionsGetEnum(NULL, NULL, "-mat_factor_type", MatFactorTypes, (PetscEnum *)&factor, NULL));
  PetscCall(MatGetFactor(A, solver, factor, &F));

  PetscCall(ISCreateStride(PETSC_COMM_SELF, ns, m - ns, 1, &is_schur));
  PetscCall(MatFactorSetSchurIS(F, is_schur));
  PetscCall(ISDestroy(&is_schur));
  switch (factor) {
  case MAT_FACTOR_LU:
    PetscCall(MatLUFactorSymbolic(F, A, NULL, NULL, NULL));
    PetscCall(MatLUFactorNumeric(F, A, NULL));
    break;
  case MAT_FACTOR_CHOLESKY:
    PetscCall(MatCholeskyFactorSymbolic(F, A, NULL, NULL));
    PetscCall(MatCholeskyFactorNumeric(F, A, NULL));
    break;
  default:
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_SUP, "Not coded for factor type %s", MatFactorTypes[factor]);
  }

  PetscCall(MatFactorCreateSchurComplement(F, &S, NULL));
  PetscCall(MatViewFromOptions(S, NULL, "-S_view"));
  PetscCall(MatDestroy(&S));

  PetscCall(MatGetSize(B, NULL, &n));
  PetscCall(MatCreate(PETSC_COMM_WORLD, &X));
  PetscCall(MatSetSizes(X, m, PETSC_DECIDE, PETSC_DECIDE, n));
  PetscCall(MatSetType(X, MATDENSE));
  PetscCall(MatSetFromOptions(X));
  PetscCall(MatSetUp(X));

  PetscCall(MatMatSolve(F, B, X));
  PetscCall(MatViewFromOptions(X, NULL, "-X_view"));
  PetscCall(MatMatMult(A, X, MAT_INITIAL_MATRIX, PETSC_DETERMINE, &Y));
  PetscCall(MatViewFromOptions(Y, NULL, "-Y_view"));
  PetscCall(MatAXPY(Y, -1.0, B, SAME_NONZERO_PATTERN));
  PetscCall(MatViewFromOptions(Y, NULL, "-err_view"));
  PetscCall(MatNorm(Y, NORM_FROBENIUS, &norm));
  if (norm > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatMatSolve: Norm of error %g\n", (double)norm));
    PetscCall(MatConvert(Y, MATAIJ, MAT_INPLACE_MATRIX, &Y));
    PetscCall(MatFilter(Y, PETSC_SMALL, PETSC_TRUE, PETSC_FALSE));
    PetscCall(MatViewFromOptions(Y, NULL, "-aij_err_view"));
  }
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&Y));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    output_file: output/empty.out
    suffix: mumps_1
    requires: datafilespath mumps double !complex !defined(PETSC_USE_64BIT_INDICES)
    args: -A ${DATAFILESPATH}/matrices/factorSchur/A.dat -B ${DATAFILESPATH}/matrices/factorSchur/B1.dat -ns {{0 1}}

  test:
    output_file: output/empty.out
    suffix: mumps_2
    requires: datafilespath mumps double !complex !defined(PETSC_USE_64BIT_INDICES)
    args: -A ${DATAFILESPATH}/matrices/factorSchur/A.dat -B ${DATAFILESPATH}/matrices/factorSchur/B2.dat -ns {{0 1}}

TEST*/
