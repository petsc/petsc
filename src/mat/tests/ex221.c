static char help[] = "Tests various routines for MATSHELL\n\n";

#include <petscmat.h>

typedef struct _n_User *User;
struct _n_User {
  Mat B;
};

static PetscErrorCode MatGetDiagonal_User(Mat A, Vec X)
{
  User user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &user));
  PetscCall(MatGetDiagonal(user->B, X));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMult_User(Mat A, Vec X, Vec Y)
{
  User user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &user));
  PetscCall(MatMult(user->B, X, Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatMultTranspose_User(Mat A, Vec X, Vec Y)
{
  User user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &user));
  PetscCall(MatMultTranspose(user->B, X, Y));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatCopy_User(Mat A, Mat X, MatStructure str)
{
  User user, userX;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &user));
  PetscCall(MatShellGetContext(X, &userX));
  PetscCheck(user == userX, PetscObjectComm((PetscObject)A), PETSC_ERR_PLIB, "This should not happen");
  PetscCall(PetscObjectReference((PetscObject)user->B));
  PetscFunctionReturn(0);
}

static PetscErrorCode MatDestroy_User(Mat A)
{
  User user;

  PetscFunctionBegin;
  PetscCall(MatShellGetContext(A, &user));
  PetscCall(PetscObjectDereference((PetscObject)user->B));
  PetscFunctionReturn(0);
}

int main(int argc, char **args)
{
  User         user;
  Mat          A, S;
  PetscScalar *data, diag = 1.3;
  PetscReal    tol = PETSC_SMALL;
  PetscInt     i, j, m = PETSC_DECIDE, n = PETSC_DECIDE, M = 17, N = 15, s1, s2;
  PetscInt     test, ntest = 2;
  PetscMPIInt  rank, size;
  PetscBool    nc        = PETSC_FALSE, cong, flg;
  PetscBool    ronl      = PETSC_TRUE;
  PetscBool    randomize = PETSC_FALSE, submat = PETSC_FALSE;
  PetscBool    keep         = PETSC_FALSE;
  PetscBool    testzerorows = PETSC_TRUE, testdiagscale = PETSC_TRUE, testgetdiag = PETSC_TRUE, testsubmat = PETSC_TRUE;
  PetscBool    testshift = PETSC_TRUE, testscale = PETSC_TRUE, testdup = PETSC_TRUE, testreset = PETSC_TRUE;
  PetscBool    testaxpy = PETSC_TRUE, testaxpyd = PETSC_TRUE, testaxpyerr = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD, &rank));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-M", &M, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-N", &N, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ml", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nl", &n, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-square_nc", &nc, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-rows_only", &ronl, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-randomize", &randomize, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-submat", &submat, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_zerorows", &testzerorows, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_diagscale", &testdiagscale, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_getdiag", &testgetdiag, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_shift", &testshift, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_scale", &testscale, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_dup", &testdup, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_reset", &testreset, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_submat", &testsubmat, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_axpy", &testaxpy, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_axpy_different", &testaxpyd, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_axpy_error", &testaxpyerr, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-loop", &ntest, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-tol", &tol, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-diag", &diag, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-keep", &keep, NULL));
  /* This tests square matrices with different row/col layout */
  if (nc && size > 1) {
    M = PetscMax(PetscMax(N, M), 1);
    N = M;
    m = n = 0;
    if (rank == 0) {
      m = M - 1;
      n = 1;
    } else if (rank == 1) {
      m = 1;
      n = N - 1;
    }
  }
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, n, M, N, NULL, &A));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetOwnershipRange(A, &s1, NULL));
  s2 = 1;
  while (s2 < M) s2 *= 10;
  PetscCall(MatDenseGetArray(A, &data));
  for (j = 0; j < N; j++) {
    for (i = 0; i < m; i++) data[j * m + i] = s2 * j + i + s1 + 1;
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  if (submat) {
    Mat      A2;
    IS       r, c;
    PetscInt rst, ren, cst, cen;

    PetscCall(MatGetOwnershipRange(A, &rst, &ren));
    PetscCall(MatGetOwnershipRangeColumn(A, &cst, &cen));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A), (ren - rst) / 2, rst, 1, &r));
    PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A), (cen - cst) / 2, cst, 1, &c));
    PetscCall(MatCreateSubMatrix(A, r, c, MAT_INITIAL_MATRIX, &A2));
    PetscCall(ISDestroy(&r));
    PetscCall(ISDestroy(&c));
    PetscCall(MatDestroy(&A));
    A = A2;
  }

  PetscCall(MatGetSize(A, &M, &N));
  PetscCall(MatGetLocalSize(A, &m, &n));
  PetscCall(MatHasCongruentLayouts(A, &cong));

  PetscCall(MatConvert(A, MATAIJ, MAT_INPLACE_MATRIX, &A));
  PetscCall(MatSetOption(A, MAT_KEEP_NONZERO_PATTERN, keep));
  PetscCall(PetscObjectSetName((PetscObject)A, "initial"));
  PetscCall(MatViewFromOptions(A, NULL, "-view_mat"));

  PetscCall(PetscNew(&user));
  PetscCall(MatCreateShell(PETSC_COMM_WORLD, m, n, M, N, user, &S));
  PetscCall(MatShellSetOperation(S, MATOP_MULT, (void (*)(void))MatMult_User));
  PetscCall(MatShellSetOperation(S, MATOP_MULT_TRANSPOSE, (void (*)(void))MatMultTranspose_User));
  if (cong) PetscCall(MatShellSetOperation(S, MATOP_GET_DIAGONAL, (void (*)(void))MatGetDiagonal_User));
  PetscCall(MatShellSetOperation(S, MATOP_COPY, (void (*)(void))MatCopy_User));
  PetscCall(MatShellSetOperation(S, MATOP_DESTROY, (void (*)(void))MatDestroy_User));
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &user->B));

  /* Square and rows only scaling */
  ronl = cong ? ronl : PETSC_TRUE;

  for (test = 0; test < ntest; test++) {
    PetscReal err;

    PetscCall(MatMultAddEqual(A, S, 10, &flg));
    if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error mult add\n", test));
    PetscCall(MatMultTransposeAddEqual(A, S, 10, &flg));
    if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error mult add (T)\n", test));
    if (testzerorows) {
      Mat       ST, B, C, BT, BTT;
      IS        zr;
      Vec       x = NULL, b1 = NULL, b2 = NULL;
      PetscInt *idxs = NULL, nr = 0;

      if (rank == (test % size)) {
        nr = 1;
        PetscCall(PetscMalloc1(nr, &idxs));
        if (test % 2) {
          idxs[0] = (2 * M - 1 - test / 2) % M;
        } else {
          idxs[0] = (test / 2) % M;
        }
        idxs[0] = PetscMax(idxs[0], 0);
      }
      PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, nr, idxs, PETSC_OWN_POINTER, &zr));
      PetscCall(PetscObjectSetName((PetscObject)zr, "ZR"));
      PetscCall(ISViewFromOptions(zr, NULL, "-view_is"));
      PetscCall(MatCreateVecs(A, &x, &b1));
      if (randomize) {
        PetscCall(VecSetRandom(x, NULL));
        PetscCall(VecSetRandom(b1, NULL));
      } else {
        PetscCall(VecSet(x, 11.4));
        PetscCall(VecSet(b1, -14.2));
      }
      PetscCall(VecDuplicate(b1, &b2));
      PetscCall(VecCopy(b1, b2));
      PetscCall(PetscObjectSetName((PetscObject)b1, "A_B1"));
      PetscCall(PetscObjectSetName((PetscObject)b2, "A_B2"));
      if (size > 1 && !cong) { /* MATMPIAIJ ZeroRows and ZeroRowsColumns are buggy in this case */
        PetscCall(VecDestroy(&b1));
      }
      if (ronl) {
        PetscCall(MatZeroRowsIS(A, zr, diag, x, b1));
        PetscCall(MatZeroRowsIS(S, zr, diag, x, b2));
      } else {
        PetscCall(MatZeroRowsColumnsIS(A, zr, diag, x, b1));
        PetscCall(MatZeroRowsColumnsIS(S, zr, diag, x, b2));
        PetscCall(ISDestroy(&zr));
        /* Mix zerorows and zerorowscols */
        nr   = 0;
        idxs = NULL;
        if (rank == 0) {
          nr = 1;
          PetscCall(PetscMalloc1(nr, &idxs));
          if (test % 2) {
            idxs[0] = (3 * M - 2 - test / 2) % M;
          } else {
            idxs[0] = (test / 2 + 1) % M;
          }
          idxs[0] = PetscMax(idxs[0], 0);
        }
        PetscCall(ISCreateGeneral(PETSC_COMM_WORLD, nr, idxs, PETSC_OWN_POINTER, &zr));
        PetscCall(PetscObjectSetName((PetscObject)zr, "ZR2"));
        PetscCall(ISViewFromOptions(zr, NULL, "-view_is"));
        PetscCall(MatZeroRowsIS(A, zr, diag * 2.0 + PETSC_SMALL, NULL, NULL));
        PetscCall(MatZeroRowsIS(S, zr, diag * 2.0 + PETSC_SMALL, NULL, NULL));
      }
      PetscCall(ISDestroy(&zr));

      if (b1) {
        Vec b;

        PetscCall(VecViewFromOptions(b1, NULL, "-view_b"));
        PetscCall(VecViewFromOptions(b2, NULL, "-view_b"));
        PetscCall(VecDuplicate(b1, &b));
        PetscCall(VecCopy(b1, b));
        PetscCall(VecAXPY(b, -1.0, b2));
        PetscCall(VecNorm(b, NORM_INFINITY, &err));
        if (err >= tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error b %g\n", test, (double)err));
        PetscCall(VecDestroy(&b));
      }
      PetscCall(VecDestroy(&b1));
      PetscCall(VecDestroy(&b2));
      PetscCall(VecDestroy(&x));
      PetscCall(MatConvert(S, MATDENSE, MAT_INITIAL_MATRIX, &B));

      PetscCall(MatCreateTranspose(S, &ST));
      PetscCall(MatComputeOperator(ST, MATDENSE, &BT));
      PetscCall(MatTranspose(BT, MAT_INITIAL_MATRIX, &BTT));
      PetscCall(PetscObjectSetName((PetscObject)B, "S"));
      PetscCall(PetscObjectSetName((PetscObject)BTT, "STT"));
      PetscCall(MatConvert(A, MATDENSE, MAT_INITIAL_MATRIX, &C));
      PetscCall(PetscObjectSetName((PetscObject)C, "A"));

      PetscCall(MatViewFromOptions(C, NULL, "-view_mat"));
      PetscCall(MatViewFromOptions(B, NULL, "-view_mat"));
      PetscCall(MatViewFromOptions(BTT, NULL, "-view_mat"));

      PetscCall(MatAXPY(C, -1.0, B, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(C, NORM_FROBENIUS, &err));
      if (err >= tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error mat mult after %s %g\n", test, ronl ? "MatZeroRows" : "MatZeroRowsColumns", (double)err));

      PetscCall(MatConvert(A, MATDENSE, MAT_REUSE_MATRIX, &C));
      PetscCall(MatAXPY(C, -1.0, BTT, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(C, NORM_FROBENIUS, &err));
      if (err >= tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error mat mult transpose after %s %g\n", test, ronl ? "MatZeroRows" : "MatZeroRowsColumns", (double)err));

      PetscCall(MatDestroy(&ST));
      PetscCall(MatDestroy(&BTT));
      PetscCall(MatDestroy(&BT));
      PetscCall(MatDestroy(&B));
      PetscCall(MatDestroy(&C));
    }
    if (testdiagscale) { /* MatDiagonalScale() */
      Vec vr, vl;

      PetscCall(MatCreateVecs(A, &vr, &vl));
      if (randomize) {
        PetscCall(VecSetRandom(vr, NULL));
        PetscCall(VecSetRandom(vl, NULL));
      } else {
        PetscCall(VecSet(vr, test % 2 ? 0.15 : 1.0 / 0.15));
        PetscCall(VecSet(vl, test % 2 ? -1.2 : 1.0 / -1.2));
      }
      PetscCall(MatDiagonalScale(A, vl, vr));
      PetscCall(MatDiagonalScale(S, vl, vr));
      PetscCall(VecDestroy(&vr));
      PetscCall(VecDestroy(&vl));
    }

    if (testscale) { /* MatScale() */
      PetscCall(MatScale(A, test % 2 ? 1.4 : 1.0 / 1.4));
      PetscCall(MatScale(S, test % 2 ? 1.4 : 1.0 / 1.4));
    }

    if (testshift && cong) { /* MatShift() : MATSHELL shift is broken when row/cols layout are not congruent and left/right scaling have been applied */
      PetscCall(MatShift(A, test % 2 ? -77.5 : 77.5));
      PetscCall(MatShift(S, test % 2 ? -77.5 : 77.5));
    }

    if (testgetdiag && cong) { /* MatGetDiagonal() */
      Vec dA, dS;

      PetscCall(MatCreateVecs(A, &dA, NULL));
      PetscCall(MatCreateVecs(S, &dS, NULL));
      PetscCall(MatGetDiagonal(A, dA));
      PetscCall(MatGetDiagonal(S, dS));
      PetscCall(VecAXPY(dA, -1.0, dS));
      PetscCall(VecNorm(dA, NORM_INFINITY, &err));
      if (err >= tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error diag %g\n", test, (double)err));
      PetscCall(VecDestroy(&dA));
      PetscCall(VecDestroy(&dS));
    }

    if (testdup && !test) {
      Mat A2, S2;

      PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &A2));
      PetscCall(MatDuplicate(S, MAT_COPY_VALUES, &S2));
      PetscCall(MatDestroy(&A));
      PetscCall(MatDestroy(&S));
      A = A2;
      S = S2;
    }

    if (testsubmat) {
      Mat      sA, sS, dA, dS, At, St;
      IS       r, c;
      PetscInt rst, ren, cst, cen;

      PetscCall(MatGetOwnershipRange(A, &rst, &ren));
      PetscCall(MatGetOwnershipRangeColumn(A, &cst, &cen));
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A), (ren - rst) / 2, rst, 1, &r));
      PetscCall(ISCreateStride(PetscObjectComm((PetscObject)A), (cen - cst) / 2, cst, 1, &c));
      PetscCall(MatCreateSubMatrix(A, r, c, MAT_INITIAL_MATRIX, &sA));
      PetscCall(MatCreateSubMatrix(S, r, c, MAT_INITIAL_MATRIX, &sS));
      PetscCall(MatMultAddEqual(sA, sS, 10, &flg));
      if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error submatrix mult add\n", test));
      PetscCall(MatMultTransposeAddEqual(sA, sS, 10, &flg));
      if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error submatrix mult add (T)\n", test));
      PetscCall(MatConvert(sA, MATDENSE, MAT_INITIAL_MATRIX, &dA));
      PetscCall(MatConvert(sS, MATDENSE, MAT_INITIAL_MATRIX, &dS));
      PetscCall(MatAXPY(dA, -1.0, dS, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(dA, NORM_FROBENIUS, &err));
      if (err >= tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error mat submatrix %g\n", test, (double)err));
      PetscCall(MatDestroy(&sA));
      PetscCall(MatDestroy(&sS));
      PetscCall(MatDestroy(&dA));
      PetscCall(MatDestroy(&dS));
      PetscCall(MatCreateTranspose(A, &At));
      PetscCall(MatCreateTranspose(S, &St));
      PetscCall(MatCreateSubMatrix(At, c, r, MAT_INITIAL_MATRIX, &sA));
      PetscCall(MatCreateSubMatrix(St, c, r, MAT_INITIAL_MATRIX, &sS));
      PetscCall(MatMultAddEqual(sA, sS, 10, &flg));
      if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error submatrix (T) mult add\n", test));
      PetscCall(MatMultTransposeAddEqual(sA, sS, 10, &flg));
      if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error submatrix (T) mult add (T)\n", test));
      PetscCall(MatConvert(sA, MATDENSE, MAT_INITIAL_MATRIX, &dA));
      PetscCall(MatConvert(sS, MATDENSE, MAT_INITIAL_MATRIX, &dS));
      PetscCall(MatAXPY(dA, -1.0, dS, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(dA, NORM_FROBENIUS, &err));
      if (err >= tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error mat submatrix (T) %g\n", test, (double)err));
      PetscCall(MatDestroy(&sA));
      PetscCall(MatDestroy(&sS));
      PetscCall(MatDestroy(&dA));
      PetscCall(MatDestroy(&dS));
      PetscCall(MatDestroy(&At));
      PetscCall(MatDestroy(&St));
      PetscCall(ISDestroy(&r));
      PetscCall(ISDestroy(&c));
    }

    if (testaxpy) {
      Mat          tA, tS, dA, dS;
      MatStructure str[3] = {SAME_NONZERO_PATTERN, SUBSET_NONZERO_PATTERN, DIFFERENT_NONZERO_PATTERN};

      PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &tA));
      if (testaxpyd && !(test % 2)) {
        PetscCall(PetscObjectReference((PetscObject)tA));
        tS = tA;
      } else {
        PetscCall(PetscObjectReference((PetscObject)S));
        tS = S;
      }
      PetscCall(MatAXPY(A, 0.5, tA, str[test % 3]));
      PetscCall(MatAXPY(S, 0.5, tS, str[test % 3]));
      /* this will trigger an error the next MatMult or MatMultTranspose call for S */
      if (testaxpyerr) PetscCall(MatScale(tA, 0));
      PetscCall(MatDestroy(&tA));
      PetscCall(MatDestroy(&tS));
      PetscCall(MatMultAddEqual(A, S, 10, &flg));
      if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error axpy mult add\n", test));
      PetscCall(MatMultTransposeAddEqual(A, S, 10, &flg));
      if (!flg) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error axpy mult add (T)\n", test));
      PetscCall(MatConvert(A, MATDENSE, MAT_INITIAL_MATRIX, &dA));
      PetscCall(MatConvert(S, MATDENSE, MAT_INITIAL_MATRIX, &dS));
      PetscCall(MatAXPY(dA, -1.0, dS, SAME_NONZERO_PATTERN));
      PetscCall(MatNorm(dA, NORM_FROBENIUS, &err));
      if (err >= tol) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "[test %" PetscInt_FMT "] Error mat submatrix %g\n", test, (double)err));
      PetscCall(MatDestroy(&dA));
      PetscCall(MatDestroy(&dS));
    }

    if (testreset && (ntest == 1 || test == ntest - 2)) {
      /* reset MATSHELL */
      PetscCall(MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY));
      PetscCall(MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY));
      /* reset A */
      PetscCall(MatCopy(user->B, A, DIFFERENT_NONZERO_PATTERN));
    }
  }

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&S));
  PetscCall(PetscFree(user));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
     suffix: rect
     requires: !single
     output_file: output/ex221_1.out
     nsize: {{1 3}}
     args: -loop 3 -keep {{0 1}} -M {{12 19}} -N {{19 12}} -submat {{0 1}} -test_axpy_different {{0 1}}

   testset:
     suffix: square
     requires: !single
     output_file: output/ex221_1.out
     nsize: {{1 3}}
     args: -M 21 -N 21 -loop 4 -rows_only {{0 1}} -keep {{0 1}} -submat {{0 1}} -test_axpy_different {{0 1}}
TEST*/
