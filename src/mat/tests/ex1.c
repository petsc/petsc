
static char help[] = "Tests LU, Cholesky, and QR factorization and MatMatSolve() for a sequential dense matrix. \n\
                      For MATSEQDENSE matrix, the factorization is just a thin wrapper to LAPACK.       \n\
                      For MATSEQDENSECUDA, it uses cusolverDn routines \n\n";

#include <petscmat.h>

static PetscErrorCode createMatsAndVecs(PetscInt m, PetscInt n, PetscInt nrhs, PetscBool full, Mat *_mat, Mat *_RHS, Mat *_SOLU, Vec *_x, Vec *_y, Vec *_b)
{
  PetscRandom rand;
  Mat         mat, RHS, SOLU;
  PetscInt    rstart, rend;
  PetscInt    cstart, cend;
  PetscScalar value = 1.0;
  Vec         x, y, b;

  PetscFunctionBegin;
  /* create multiple vectors RHS and SOLU */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &RHS));
  PetscCall(MatSetSizes(RHS, PETSC_DECIDE, PETSC_DECIDE, m, nrhs));
  PetscCall(MatSetType(RHS, MATDENSE));
  PetscCall(MatSetOptionsPrefix(RHS, "rhs_"));
  PetscCall(MatSetFromOptions(RHS));
  PetscCall(MatSeqDenseSetPreallocation(RHS, NULL));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(MatSetRandom(RHS, rand));

  if (m == n) {
    PetscCall(MatDuplicate(RHS, MAT_DO_NOT_COPY_VALUES, &SOLU));
  } else {
    PetscCall(MatCreate(PETSC_COMM_WORLD, &SOLU));
    PetscCall(MatSetSizes(SOLU, PETSC_DECIDE, PETSC_DECIDE, n, nrhs));
    PetscCall(MatSetType(SOLU, MATDENSE));
    PetscCall(MatSeqDenseSetPreallocation(SOLU, NULL));
  }
  PetscCall(MatSetRandom(SOLU, rand));

  /* create matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &mat));
  PetscCall(MatSetSizes(mat, PETSC_DECIDE, PETSC_DECIDE, m, n));
  PetscCall(MatSetType(mat, MATDENSE));
  PetscCall(MatSetFromOptions(mat));
  PetscCall(MatSetUp(mat));
  PetscCall(MatGetOwnershipRange(mat, &rstart, &rend));
  PetscCall(MatGetOwnershipRangeColumn(mat, &cstart, &cend));
  if (!full) {
    for (PetscInt i = rstart; i < rend; i++) {
      if (m == n) {
        value = (PetscReal)i + 1;
        PetscCall(MatSetValues(mat, 1, &i, 1, &i, &value, INSERT_VALUES));
      } else {
        for (PetscInt j = cstart; j < cend; j++) {
          value = ((PetscScalar)i + 1.) / (PetscSqr(i - j) + 1.);
          PetscCall(MatSetValues(mat, 1, &i, 1, &j, &value, INSERT_VALUES));
        }
      }
    }
    PetscCall(MatAssemblyBegin(mat, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(mat, MAT_FINAL_ASSEMBLY));
  } else {
    PetscCall(MatSetRandom(mat, rand));
    if (m == n) {
      Mat T;

      PetscCall(MatMatTransposeMult(mat, mat, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &T));
      PetscCall(MatDestroy(&mat));
      mat = T;
    }
  }

  /* create single vectors */
  PetscCall(MatCreateVecs(mat, &x, &b));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecSet(x, value));
  PetscCall(PetscRandomDestroy(&rand));
  *_mat  = mat;
  *_RHS  = RHS;
  *_SOLU = SOLU;
  *_x    = x;
  *_y    = y;
  *_b    = b;
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat         mat, F, RHS, SOLU;
  MatInfo     info;
  PetscInt    m = 15, n = 10, i, j, nrhs = 2;
  Vec         x, y, b, ytmp;
  IS          perm;
  PetscReal   norm, tol = PETSC_SMALL;
  PetscMPIInt size;
  char        solver[64];
  PetscBool   inplace, full = PETSC_FALSE, ldl = PETSC_TRUE, qr = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
  PetscCall(PetscStrncpy(solver, MATSOLVERPETSC, sizeof(solver)));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nrhs", &nrhs, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ldl", &ldl, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-qr", &qr, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-full", &full, NULL));
  PetscCall(PetscOptionsGetString(NULL, NULL, "-solver_type", solver, sizeof(solver), NULL));

  PetscCall(createMatsAndVecs(n, n, nrhs, full, &mat, &RHS, &SOLU, &x, &y, &b));
  PetscCall(VecDuplicate(y, &ytmp));

  /* Only SeqDense* support in-place factorizations and NULL permutations */
  PetscCall(PetscObjectBaseTypeCompare((PetscObject)mat, MATSEQDENSE, &inplace));
  PetscCall(MatGetLocalSize(mat, &i, NULL));
  PetscCall(MatGetOwnershipRange(mat, &j, NULL));
  PetscCall(ISCreateStride(PETSC_COMM_WORLD, i, j, 1, &perm));

  PetscCall(MatGetInfo(mat, MAT_LOCAL, &info));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "matrix nonzeros = %" PetscInt_FMT ", allocated nonzeros = %" PetscInt_FMT "\n", (PetscInt)info.nz_used, (PetscInt)info.nz_allocated));
  PetscCall(MatMult(mat, x, b));

  /* Cholesky factorization - perm and factinfo are ignored by LAPACK */
  /* in-place Cholesky */
  if (inplace) {
    Mat RHS2;

    PetscCall(MatDuplicate(mat, MAT_COPY_VALUES, &F));
    if (!ldl) PetscCall(MatSetOption(F, MAT_SPD, PETSC_TRUE));
    PetscCall(MatCholeskyFactor(F, perm, 0));
    PetscCall(MatSolve(F, b, y));
    PetscCall(VecAXPY(y, -1.0, x));
    PetscCall(VecNorm(y, NORM_2, &norm));
    if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Norm of error for in-place Cholesky %g\n", (double)norm));

    PetscCall(MatMatSolve(F, RHS, SOLU));
    PetscCall(MatMatMult(mat, SOLU, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RHS2));
    PetscCall(MatAXPY(RHS, -1.0, RHS2, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(RHS, NORM_FROBENIUS, &norm));
    if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: Norm of residual for in-place Cholesky (MatMatSolve) %g\n", (double)norm));
    PetscCall(MatDestroy(&F));
    PetscCall(MatDestroy(&RHS2));
  }

  /* out-of-place Cholesky */
  PetscCall(MatGetFactor(mat, solver, MAT_FACTOR_CHOLESKY, &F));
  if (!ldl) PetscCall(MatSetOption(F, MAT_SPD, PETSC_TRUE));
  PetscCall(MatCholeskyFactorSymbolic(F, mat, perm, 0));
  PetscCall(MatCholeskyFactorNumeric(F, mat, 0));
  PetscCall(MatSolve(F, b, y));
  PetscCall(VecAXPY(y, -1.0, x));
  PetscCall(VecNorm(y, NORM_2, &norm));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Norm of error for out-of-place Cholesky %g\n", (double)norm));
  PetscCall(MatDestroy(&F));

  /* LU factorization - perms and factinfo are ignored by LAPACK */
  i = n - 1;
  PetscCall(MatZeroRows(mat, 1, &i, -1.0, NULL, NULL));
  PetscCall(MatMult(mat, x, b));

  /* in-place LU */
  if (inplace) {
    Mat RHS2;

    PetscCall(MatDuplicate(mat, MAT_COPY_VALUES, &F));
    PetscCall(MatLUFactor(F, perm, perm, 0));
    PetscCall(MatSolve(F, b, y));
    PetscCall(VecAXPY(y, -1.0, x));
    PetscCall(VecNorm(y, NORM_2, &norm));
    if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Norm of error for in-place LU %g\n", (double)norm));
    PetscCall(MatMatSolve(F, RHS, SOLU));
    PetscCall(MatMatMult(mat, SOLU, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &RHS2));
    PetscCall(MatAXPY(RHS, -1.0, RHS2, SAME_NONZERO_PATTERN));
    PetscCall(MatNorm(RHS, NORM_FROBENIUS, &norm));
    if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: Norm of residual for in-place LU (MatMatSolve) %g\n", (double)norm));
    PetscCall(MatDestroy(&F));
    PetscCall(MatDestroy(&RHS2));
  }

  /* out-of-place LU */
  PetscCall(MatGetFactor(mat, solver, MAT_FACTOR_LU, &F));
  PetscCall(MatLUFactorSymbolic(F, mat, perm, perm, 0));
  PetscCall(MatLUFactorNumeric(F, mat, 0));
  PetscCall(MatSolve(F, b, y));
  PetscCall(VecAXPY(y, -1.0, x));
  PetscCall(VecNorm(y, NORM_2, &norm));
  if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Norm of error for out-of-place LU %g\n", (double)norm));

  /* free space */
  PetscCall(ISDestroy(&perm));
  PetscCall(MatDestroy(&F));
  PetscCall(MatDestroy(&mat));
  PetscCall(MatDestroy(&RHS));
  PetscCall(MatDestroy(&SOLU));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&ytmp));

  if (qr) {
    /* setup rectangular */
    PetscCall(createMatsAndVecs(m, n, nrhs, full, &mat, &RHS, &SOLU, &x, &y, &b));
    PetscCall(VecDuplicate(y, &ytmp));

    /* QR factorization - perms and factinfo are ignored by LAPACK */
    PetscCall(MatMult(mat, x, b));

    /* in-place QR */
    if (inplace) {
      Mat SOLU2;

      PetscCall(MatDuplicate(mat, MAT_COPY_VALUES, &F));
      PetscCall(MatQRFactor(F, NULL, 0));
      PetscCall(MatSolve(F, b, y));
      PetscCall(VecAXPY(y, -1.0, x));
      PetscCall(VecNorm(y, NORM_2, &norm));
      if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Norm of error for in-place QR %g\n", (double)norm));
      PetscCall(MatMatMult(mat, SOLU, MAT_REUSE_MATRIX, PETSC_DEFAULT, &RHS));
      PetscCall(MatDuplicate(SOLU, MAT_DO_NOT_COPY_VALUES, &SOLU2));
      PetscCall(MatMatSolve(F, RHS, SOLU2));
      PetscCall(MatAXPY(SOLU2, -1.0, SOLU, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(SOLU2, NORM_FROBENIUS, &norm));
      if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error: Norm of error for in-place QR (MatMatSolve) %g\n", (double)norm));
      PetscCall(MatDestroy(&F));
      PetscCall(MatDestroy(&SOLU2));
    }

    /* out-of-place QR */
    PetscCall(MatGetFactor(mat, solver, MAT_FACTOR_QR, &F));
    PetscCall(MatQRFactorSymbolic(F, mat, NULL, NULL));
    PetscCall(MatQRFactorNumeric(F, mat, NULL));
    PetscCall(MatSolve(F, b, y));
    PetscCall(VecAXPY(y, -1.0, x));
    PetscCall(VecNorm(y, NORM_2, &norm));
    if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Norm of error for out-of-place QR %g\n", (double)norm));

    if (m == n) {
      /* out-of-place MatSolveTranspose */
      PetscCall(MatMultTranspose(mat, x, b));
      PetscCall(MatSolveTranspose(F, b, y));
      PetscCall(VecAXPY(y, -1.0, x));
      PetscCall(VecNorm(y, NORM_2, &norm));
      if (norm > tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Warning: Norm of error for out-of-place QR %g\n", (double)norm));
    }

    /* free space */
    PetscCall(MatDestroy(&F));
    PetscCall(MatDestroy(&mat));
    PetscCall(MatDestroy(&RHS));
    PetscCall(MatDestroy(&SOLU));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&y));
    PetscCall(VecDestroy(&ytmp));
  }
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
     requires: cuda
     suffix: seqdensecuda
     args: -mat_type seqdensecuda -rhs_mat_type seqdensecuda -ldl 0 -solver_type {{petsc cuda}}
     output_file: output/ex1_1.out

   test:
     requires: cuda
     suffix: seqdensecuda_2
     args: -ldl 0 -solver_type cuda
     output_file: output/ex1_1.out

   test:
     requires: cuda
     suffix: seqdensecuda_seqaijcusparse
     args: -mat_type seqaijcusparse -rhs_mat_type seqdensecuda -qr 0
     output_file: output/ex1_2.out

   test:
     requires: cuda viennacl
     suffix: seqdensecuda_seqaijviennacl
     args: -mat_type seqaijviennacl -rhs_mat_type seqdensecuda -qr 0
     output_file: output/ex1_2.out

   test:
     suffix: 4
     args: -m 10 -n 10
     output_file: output/ex1_1.out

TEST*/
