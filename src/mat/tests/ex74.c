
static char help[] = "Tests the various sequential routines in MATSEQSBAIJ format.\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  PetscMPIInt   size;
  Vec           x, y, b, s1, s2;
  Mat           A;                     /* linear system matrix */
  Mat           sA, sB, sFactor, B, C; /* symmetric matrices */
  PetscInt      n, mbs = 16, bs = 1, nz = 3, prob = 1, i, j, k1, k2, col[3], lf, block, row, Ii, J, n1, inc;
  PetscReal     norm1, norm2, rnorm, tol = 10 * PETSC_SMALL;
  PetscScalar   neg_one = -1.0, four = 4.0, value[3];
  IS            perm, iscol;
  PetscRandom   rdm;
  PetscBool     doIcc = PETSC_TRUE, equal;
  MatInfo       minfo1, minfo2;
  MatFactorInfo factinfo;
  MatType       type;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-bs", &bs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mbs", &mbs, NULL));

  n = mbs * bs;
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(A, MATSEQBAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqBAIJSetPreallocation(A, bs, nz, NULL));

  PetscCall(MatCreate(PETSC_COMM_SELF, &sA));
  PetscCall(MatSetSizes(sA, n, n, PETSC_DETERMINE, PETSC_DETERMINE));
  PetscCall(MatSetType(sA, MATSEQSBAIJ));
  PetscCall(MatSetFromOptions(sA));
  PetscCall(MatGetType(sA, &type));
  PetscCall(PetscObjectTypeCompare((PetscObject)sA, MATSEQSBAIJ, &doIcc));
  PetscCall(MatSeqSBAIJSetPreallocation(sA, bs, nz, NULL));
  PetscCall(MatSetOption(sA, MAT_IGNORE_LOWER_TRIANGULAR, PETSC_TRUE));

  /* Test MatGetOwnershipRange() */
  PetscCall(MatGetOwnershipRange(A, &Ii, &J));
  PetscCall(MatGetOwnershipRange(sA, &i, &j));
  if (i - Ii || j - J) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error: MatGetOwnershipRange() in MatSBAIJ format\n"));

  /* Assemble matrix */
  if (bs == 1) {
    PetscCall(PetscOptionsGetInt(NULL, NULL, "-test_problem", &prob, NULL));
    if (prob == 1) { /* tridiagonal matrix */
      value[0] = -1.0;
      value[1] = 2.0;
      value[2] = -1.0;
      for (i = 1; i < n - 1; i++) {
        col[0] = i - 1;
        col[1] = i;
        col[2] = i + 1;
        PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
        PetscCall(MatSetValues(sA, 1, &i, 3, col, value, INSERT_VALUES));
      }
      i      = n - 1;
      col[0] = 0;
      col[1] = n - 2;
      col[2] = n - 1;

      value[0] = 0.1;
      value[1] = -1;
      value[2] = 2;

      PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
      PetscCall(MatSetValues(sA, 1, &i, 3, col, value, INSERT_VALUES));

      i        = 0;
      col[0]   = n - 1;
      col[1]   = 1;
      col[2]   = 0;
      value[0] = 0.1;
      value[1] = -1.0;
      value[2] = 2;

      PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
      PetscCall(MatSetValues(sA, 1, &i, 3, col, value, INSERT_VALUES));

    } else if (prob == 2) { /* matrix for the five point stencil */
      n1 = (PetscInt)(PetscSqrtReal((PetscReal)n) + 0.001);
      PetscCheck(n1 * n1 == n, PETSC_COMM_SELF, PETSC_ERR_ARG_WRONG, "sqrt(n) must be a positive integer!");
      for (i = 0; i < n1; i++) {
        for (j = 0; j < n1; j++) {
          Ii = j + n1 * i;
          if (i > 0) {
            J = Ii - n1;
            PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &neg_one, INSERT_VALUES));
            PetscCall(MatSetValues(sA, 1, &Ii, 1, &J, &neg_one, INSERT_VALUES));
          }
          if (i < n1 - 1) {
            J = Ii + n1;
            PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &neg_one, INSERT_VALUES));
            PetscCall(MatSetValues(sA, 1, &Ii, 1, &J, &neg_one, INSERT_VALUES));
          }
          if (j > 0) {
            J = Ii - 1;
            PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &neg_one, INSERT_VALUES));
            PetscCall(MatSetValues(sA, 1, &Ii, 1, &J, &neg_one, INSERT_VALUES));
          }
          if (j < n1 - 1) {
            J = Ii + 1;
            PetscCall(MatSetValues(A, 1, &Ii, 1, &J, &neg_one, INSERT_VALUES));
            PetscCall(MatSetValues(sA, 1, &Ii, 1, &J, &neg_one, INSERT_VALUES));
          }
          PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &four, INSERT_VALUES));
          PetscCall(MatSetValues(sA, 1, &Ii, 1, &Ii, &four, INSERT_VALUES));
        }
      }
    }

  } else { /* bs > 1 */
    for (block = 0; block < n / bs; block++) {
      /* diagonal blocks */
      value[0] = -1.0;
      value[1] = 4.0;
      value[2] = -1.0;
      for (i = 1 + block * bs; i < bs - 1 + block * bs; i++) {
        col[0] = i - 1;
        col[1] = i;
        col[2] = i + 1;
        PetscCall(MatSetValues(A, 1, &i, 3, col, value, INSERT_VALUES));
        PetscCall(MatSetValues(sA, 1, &i, 3, col, value, INSERT_VALUES));
      }
      i      = bs - 1 + block * bs;
      col[0] = bs - 2 + block * bs;
      col[1] = bs - 1 + block * bs;

      value[0] = -1.0;
      value[1] = 4.0;

      PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
      PetscCall(MatSetValues(sA, 1, &i, 2, col, value, INSERT_VALUES));

      i      = 0 + block * bs;
      col[0] = 0 + block * bs;
      col[1] = 1 + block * bs;

      value[0] = 4.0;
      value[1] = -1.0;

      PetscCall(MatSetValues(A, 1, &i, 2, col, value, INSERT_VALUES));
      PetscCall(MatSetValues(sA, 1, &i, 2, col, value, INSERT_VALUES));
    }
    /* off-diagonal blocks */
    value[0] = -1.0;
    for (i = 0; i < (n / bs - 1) * bs; i++) {
      col[0] = i + bs;

      PetscCall(MatSetValues(A, 1, &i, 1, col, value, INSERT_VALUES));
      PetscCall(MatSetValues(sA, 1, &i, 1, col, value, INSERT_VALUES));

      col[0] = i;
      row    = i + bs;

      PetscCall(MatSetValues(A, 1, &row, 1, col, value, INSERT_VALUES));
      PetscCall(MatSetValues(sA, 1, &row, 1, col, value, INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatAssemblyBegin(sA, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(sA, MAT_FINAL_ASSEMBLY));

  /* Test MatGetInfo() of A and sA */
  PetscCall(MatGetInfo(A, MAT_LOCAL, &minfo1));
  PetscCall(MatGetInfo(sA, MAT_LOCAL, &minfo2));
  i  = (int)(minfo1.nz_used - minfo2.nz_used);
  j  = (int)(minfo1.nz_allocated - minfo2.nz_allocated);
  k1 = (int)(minfo1.nz_allocated - minfo1.nz_used);
  k2 = (int)(minfo2.nz_allocated - minfo2.nz_used);
  if (i < 0 || j < 0 || k1 < 0 || k2 < 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error (compare A and sA): MatGetInfo()\n"));

  /* Test MatDuplicate() */
  PetscCall(MatNorm(A, NORM_FROBENIUS, &norm1));
  PetscCall(MatDuplicate(sA, MAT_COPY_VALUES, &sB));
  PetscCall(MatEqual(sA, sB, &equal));
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "Error in MatDuplicate()");

  /* Test MatNorm() */
  PetscCall(MatNorm(A, NORM_FROBENIUS, &norm1));
  PetscCall(MatNorm(sB, NORM_FROBENIUS, &norm2));
  rnorm = PetscAbsReal(norm1 - norm2) / norm2;
  if (rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error: MatNorm_FROBENIUS, NormA=%16.14e NormsB=%16.14e\n", (double)norm1, (double)norm2));
  PetscCall(MatNorm(A, NORM_INFINITY, &norm1));
  PetscCall(MatNorm(sB, NORM_INFINITY, &norm2));
  rnorm = PetscAbsReal(norm1 - norm2) / norm2;
  if (rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error: MatNorm_INFINITY(), NormA=%16.14e NormsB=%16.14e\n", (double)norm1, (double)norm2));
  PetscCall(MatNorm(A, NORM_1, &norm1));
  PetscCall(MatNorm(sB, NORM_1, &norm2));
  rnorm = PetscAbsReal(norm1 - norm2) / norm2;
  if (rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error: MatNorm_INFINITY(), NormA=%16.14e NormsB=%16.14e\n", (double)norm1, (double)norm2));

  /* Test MatGetInfo(), MatGetSize(), MatGetBlockSize() */
  PetscCall(MatGetInfo(A, MAT_LOCAL, &minfo1));
  PetscCall(MatGetInfo(sB, MAT_LOCAL, &minfo2));
  i  = (int)(minfo1.nz_used - minfo2.nz_used);
  j  = (int)(minfo1.nz_allocated - minfo2.nz_allocated);
  k1 = (int)(minfo1.nz_allocated - minfo1.nz_used);
  k2 = (int)(minfo2.nz_allocated - minfo2.nz_used);
  if (i < 0 || j < 0 || k1 < 0 || k2 < 0) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error(compare A and sB): MatGetInfo()\n"));

  PetscCall(MatGetSize(A, &Ii, &J));
  PetscCall(MatGetSize(sB, &i, &j));
  if (i - Ii || j - J) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error: MatGetSize()\n"));

  PetscCall(MatGetBlockSize(A, &Ii));
  PetscCall(MatGetBlockSize(sB, &i));
  if (i - Ii) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error: MatGetBlockSize()\n"));

  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(VecCreateSeq(PETSC_COMM_SELF, n, &x));
  PetscCall(VecDuplicate(x, &s1));
  PetscCall(VecDuplicate(x, &s2));
  PetscCall(VecDuplicate(x, &y));
  PetscCall(VecDuplicate(x, &b));
  PetscCall(VecSetRandom(x, rdm));

  /* Test MatDiagonalScale(), MatGetDiagonal(), MatScale() */
#if !defined(PETSC_USE_COMPLEX)
  /* Scaling matrix with complex numbers results non-spd matrix,
     causing crash of MatForwardSolve() and MatBackwardSolve() */
  PetscCall(MatDiagonalScale(A, x, x));
  PetscCall(MatDiagonalScale(sB, x, x));
  PetscCall(MatMultEqual(A, sB, 10, &equal));
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_ARG_NOTSAMETYPE, "Error in MatDiagonalScale");

  PetscCall(MatGetDiagonal(A, s1));
  PetscCall(MatGetDiagonal(sB, s2));
  PetscCall(VecAXPY(s2, neg_one, s1));
  PetscCall(VecNorm(s2, NORM_1, &norm1));
  if (norm1 > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error:MatGetDiagonal(), ||s1-s2||=%g\n", (double)norm1));

  {
    PetscScalar alpha = 0.1;
    PetscCall(MatScale(A, alpha));
    PetscCall(MatScale(sB, alpha));
  }
#endif

  /* Test MatGetRowMaxAbs() */
  PetscCall(MatGetRowMaxAbs(A, s1, NULL));
  PetscCall(MatGetRowMaxAbs(sB, s2, NULL));
  PetscCall(VecNorm(s1, NORM_1, &norm1));
  PetscCall(VecNorm(s2, NORM_1, &norm2));
  norm1 -= norm2;
  if (norm1 < -tol || norm1 > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error:MatGetRowMaxAbs() \n"));

  /* Test MatMult() */
  for (i = 0; i < 40; i++) {
    PetscCall(VecSetRandom(x, rdm));
    PetscCall(MatMult(A, x, s1));
    PetscCall(MatMult(sB, x, s2));
    PetscCall(VecNorm(s1, NORM_1, &norm1));
    PetscCall(VecNorm(s2, NORM_1, &norm2));
    norm1 -= norm2;
    if (norm1 < -tol || norm1 > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error: MatMult(), norm1-norm2: %g\n", (double)norm1));
  }

  /* MatMultAdd() */
  for (i = 0; i < 40; i++) {
    PetscCall(VecSetRandom(x, rdm));
    PetscCall(VecSetRandom(y, rdm));
    PetscCall(MatMultAdd(A, x, y, s1));
    PetscCall(MatMultAdd(sB, x, y, s2));
    PetscCall(VecNorm(s1, NORM_1, &norm1));
    PetscCall(VecNorm(s2, NORM_1, &norm2));
    norm1 -= norm2;
    if (norm1 < -tol || norm1 > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error:MatMultAdd(), norm1-norm2: %g\n", (double)norm1));
  }

  /* Test MatMatMult() for sbaij and dense matrices */
  PetscCall(MatCreateSeqDense(PETSC_COMM_SELF, n, 5 * n, NULL, &B));
  PetscCall(MatSetRandom(B, rdm));
  PetscCall(MatMatMult(sA, B, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &C));
  PetscCall(MatMatMultEqual(sA, B, C, 5 * n, &equal));
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error: MatMatMult()");
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&B));

  /* Test MatCholeskyFactor(), MatICCFactor() with natural ordering */
  PetscCall(MatGetOrdering(A, MATORDERINGNATURAL, &perm, &iscol));
  PetscCall(ISDestroy(&iscol));
  norm1 = tol;
  inc   = bs;

  /* initialize factinfo */
  PetscCall(PetscMemzero(&factinfo, sizeof(MatFactorInfo)));

  for (lf = -1; lf < 10; lf += inc) {
    if (lf == -1) { /* Cholesky factor of sB (duplicate sA) */
      factinfo.fill = 5.0;

      PetscCall(MatGetFactor(sB, MATSOLVERPETSC, MAT_FACTOR_CHOLESKY, &sFactor));
      PetscCall(MatCholeskyFactorSymbolic(sFactor, sB, perm, &factinfo));
    } else if (!doIcc) break;
    else { /* incomplete Cholesky factor */ factinfo.fill = 5.0;
      factinfo.levels                                     = lf;

      PetscCall(MatGetFactor(sB, MATSOLVERPETSC, MAT_FACTOR_ICC, &sFactor));
      PetscCall(MatICCFactorSymbolic(sFactor, sB, perm, &factinfo));
    }
    PetscCall(MatCholeskyFactorNumeric(sFactor, sB, &factinfo));
    /* MatView(sFactor, PETSC_VIEWER_DRAW_WORLD); */

    /* test MatGetDiagonal on numeric factor */
    /*
    if (lf == -1) {
      PetscCall(MatGetDiagonal(sFactor,s1));
      printf(" in ex74.c, diag: \n");
      PetscCall(VecView(s1,PETSC_VIEWER_STDOUT_SELF));
    }
    */

    PetscCall(MatMult(sB, x, b));

    /* test MatForwardSolve() and MatBackwardSolve() */
    if (lf == -1) {
      PetscCall(MatForwardSolve(sFactor, b, s1));
      PetscCall(MatBackwardSolve(sFactor, s1, s2));
      PetscCall(VecAXPY(s2, neg_one, x));
      PetscCall(VecNorm(s2, NORM_2, &norm2));
      if (10 * norm1 < norm2) PetscCall(PetscPrintf(PETSC_COMM_SELF, "MatForwardSolve and BackwardSolve: Norm of error=%g, bs=%" PetscInt_FMT "\n", (double)norm2, bs));
    }

    /* test MatSolve() */
    PetscCall(MatSolve(sFactor, b, y));
    PetscCall(MatDestroy(&sFactor));
    /* Check the error */
    PetscCall(VecAXPY(y, neg_one, x));
    PetscCall(VecNorm(y, NORM_2, &norm2));
    if (10 * norm1 < norm2 && lf - inc != -1) PetscCall(PetscPrintf(PETSC_COMM_SELF, "lf=%" PetscInt_FMT ", %" PetscInt_FMT ", Norm of error=%g, %g\n", lf - inc, lf, (double)norm1, (double)norm2));
    norm1 = norm2;
    if (norm2 < tol && lf != -1) break;
  }

#if defined(PETSC_HAVE_MUMPS)
  PetscCall(MatGetFactor(sA, MATSOLVERMUMPS, MAT_FACTOR_CHOLESKY, &sFactor));
  PetscCall(MatCholeskyFactorSymbolic(sFactor, sA, NULL, NULL));
  PetscCall(MatCholeskyFactorNumeric(sFactor, sA, NULL));
  for (i = 0; i < 10; i++) {
    PetscCall(VecSetRandom(b, rdm));
    PetscCall(MatSolve(sFactor, b, y));
    /* Check the error */
    PetscCall(MatMult(sA, y, x));
    PetscCall(VecAXPY(x, neg_one, b));
    PetscCall(VecNorm(x, NORM_2, &norm2));
    if (norm2 > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error:MatSolve(), norm2: %g\n", (double)norm2));
  }
  PetscCall(MatDestroy(&sFactor));
#endif

  PetscCall(ISDestroy(&perm));

  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&sB));
  PetscCall(MatDestroy(&sA));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&s1));
  PetscCall(VecDestroy(&s2));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscRandomDestroy(&rdm));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -bs {{1 2 3 4 5 6 7 8}}

TEST*/
