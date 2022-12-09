
static char help[] = "Tests MatIncreaseOverlap(), MatCreateSubMatrices() for MatBAIJ format.\n";

#include <petscmat.h>

int main(int argc, char **args)
{
  Mat          A, B, E, Bt, *submatA, *submatB;
  PetscInt     bs = 1, m = 43, ov = 1, i, j, k, *rows, *cols, M, nd = 5, *idx, mm, nn, lsize;
  PetscScalar *vals, rval;
  IS          *is1, *is2;
  PetscRandom  rdm;
  Vec          xx, s1, s2;
  PetscReal    s1norm, s2norm, rnorm, tol = PETSC_SQRT_MACHINE_EPSILON;
  PetscBool    flg;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_block_size", &bs, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-mat_size", &m, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-ov", &ov, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-nd", &nd, NULL));
  M = m * bs;

  PetscCall(MatCreateSeqBAIJ(PETSC_COMM_SELF, bs, M, M, 1, NULL, &A));
  PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF, M, M, 15, NULL, &B));
  PetscCall(MatSetBlockSize(B, bs));
  PetscCall(MatSetOption(B, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF, &rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));

  PetscCall(PetscMalloc1(bs, &rows));
  PetscCall(PetscMalloc1(bs, &cols));
  PetscCall(PetscMalloc1(bs * bs, &vals));
  PetscCall(PetscMalloc1(M, &idx));

  /* Now set blocks of values */
  for (i = 0; i < 20 * bs; i++) {
    PetscInt nr = 1, nc = 1;
    PetscCall(PetscRandomGetValue(rdm, &rval));
    cols[0] = bs * (int)(PetscRealPart(rval) * m);
    PetscCall(PetscRandomGetValue(rdm, &rval));
    rows[0] = bs * (int)(PetscRealPart(rval) * m);
    for (j = 1; j < bs; j++) {
      PetscCall(PetscRandomGetValue(rdm, &rval));
      if (PetscRealPart(rval) > .5) rows[nr++] = rows[0] + j - 1;
    }
    for (j = 1; j < bs; j++) {
      PetscCall(PetscRandomGetValue(rdm, &rval));
      if (PetscRealPart(rval) > .5) cols[nc++] = cols[0] + j - 1;
    }

    for (j = 0; j < nr * nc; j++) {
      PetscCall(PetscRandomGetValue(rdm, &rval));
      vals[j] = rval;
    }
    PetscCall(MatSetValues(A, nr, rows, nc, cols, vals, ADD_VALUES));
    PetscCall(MatSetValues(B, nr, rows, nc, cols, vals, ADD_VALUES));
  }

  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));

  /* Test MatConvert_SeqAIJ_Seq(S)BAIJ handles incompletely filled blocks */
  PetscCall(MatConvert(B, MATBAIJ, MAT_INITIAL_MATRIX, &E));
  PetscCall(MatDestroy(&E));
  PetscCall(MatTranspose(B, MAT_INITIAL_MATRIX, &Bt));
  PetscCall(MatAXPY(Bt, 1.0, B, DIFFERENT_NONZERO_PATTERN));
  PetscCall(MatSetOption(Bt, MAT_SYMMETRIC, PETSC_TRUE));
  PetscCall(MatConvert(Bt, MATSBAIJ, MAT_INITIAL_MATRIX, &E));
  PetscCall(MatDestroy(&E));
  PetscCall(MatDestroy(&Bt));

  /* Test MatIncreaseOverlap() */
  PetscCall(PetscMalloc1(nd, &is1));
  PetscCall(PetscMalloc1(nd, &is2));

  for (i = 0; i < nd; i++) {
    PetscCall(PetscRandomGetValue(rdm, &rval));
    lsize = (int)(PetscRealPart(rval) * m);
    for (j = 0; j < lsize; j++) {
      PetscCall(PetscRandomGetValue(rdm, &rval));
      idx[j * bs] = bs * (int)(PetscRealPart(rval) * m);
      for (k = 1; k < bs; k++) idx[j * bs + k] = idx[j * bs] + k;
    }
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, lsize * bs, idx, PETSC_COPY_VALUES, is1 + i));
    PetscCall(ISCreateGeneral(PETSC_COMM_SELF, lsize * bs, idx, PETSC_COPY_VALUES, is2 + i));
  }
  PetscCall(MatIncreaseOverlap(A, nd, is1, ov));
  PetscCall(MatIncreaseOverlap(B, nd, is2, ov));

  for (i = 0; i < nd; ++i) {
    PetscCall(ISEqual(is1[i], is2[i], &flg));
    PetscCheck(flg, PETSC_COMM_SELF, PETSC_ERR_PLIB, "i=%" PetscInt_FMT ", flg =%d", i, (int)flg);
  }

  for (i = 0; i < nd; ++i) {
    PetscCall(ISSort(is1[i]));
    PetscCall(ISSort(is2[i]));
  }

  PetscCall(MatCreateSubMatrices(A, nd, is1, is1, MAT_INITIAL_MATRIX, &submatA));
  PetscCall(MatCreateSubMatrices(B, nd, is2, is2, MAT_INITIAL_MATRIX, &submatB));

  /* Test MatMult() */
  for (i = 0; i < nd; i++) {
    PetscCall(MatGetSize(submatA[i], &mm, &nn));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, mm, &xx));
    PetscCall(VecDuplicate(xx, &s1));
    PetscCall(VecDuplicate(xx, &s2));
    for (j = 0; j < 3; j++) {
      PetscCall(VecSetRandom(xx, rdm));
      PetscCall(MatMult(submatA[i], xx, s1));
      PetscCall(MatMult(submatB[i], xx, s2));
      PetscCall(VecNorm(s1, NORM_2, &s1norm));
      PetscCall(VecNorm(s2, NORM_2, &s2norm));
      rnorm = s2norm - s1norm;
      if (rnorm < -tol || rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n", (double)s1norm, (double)s2norm));
    }
    PetscCall(VecDestroy(&xx));
    PetscCall(VecDestroy(&s1));
    PetscCall(VecDestroy(&s2));
  }
  /* Now test MatCreateSubmatrices with MAT_REUSE_MATRIX option */
  PetscCall(MatCreateSubMatrices(A, nd, is1, is1, MAT_REUSE_MATRIX, &submatA));
  PetscCall(MatCreateSubMatrices(B, nd, is2, is2, MAT_REUSE_MATRIX, &submatB));

  /* Test MatMult() */
  for (i = 0; i < nd; i++) {
    PetscCall(MatGetSize(submatA[i], &mm, &nn));
    PetscCall(VecCreateSeq(PETSC_COMM_SELF, mm, &xx));
    PetscCall(VecDuplicate(xx, &s1));
    PetscCall(VecDuplicate(xx, &s2));
    for (j = 0; j < 3; j++) {
      PetscCall(VecSetRandom(xx, rdm));
      PetscCall(MatMult(submatA[i], xx, s1));
      PetscCall(MatMult(submatB[i], xx, s2));
      PetscCall(VecNorm(s1, NORM_2, &s1norm));
      PetscCall(VecNorm(s2, NORM_2, &s2norm));
      rnorm = s2norm - s1norm;
      if (rnorm < -tol || rnorm > tol) PetscCall(PetscPrintf(PETSC_COMM_SELF, "Error:MatMult - Norm1=%16.14e Norm2=%16.14e\n", (double)s1norm, (double)s2norm));
    }
    PetscCall(VecDestroy(&xx));
    PetscCall(VecDestroy(&s1));
    PetscCall(VecDestroy(&s2));
  }

  /* Free allocated memory */
  for (i = 0; i < nd; ++i) {
    PetscCall(ISDestroy(&is1[i]));
    PetscCall(ISDestroy(&is2[i]));
  }
  PetscCall(MatDestroySubMatrices(nd, &submatA));
  PetscCall(MatDestroySubMatrices(nd, &submatB));
  PetscCall(PetscFree(is1));
  PetscCall(PetscFree(is2));
  PetscCall(PetscFree(idx));
  PetscCall(PetscFree(rows));
  PetscCall(PetscFree(cols));
  PetscCall(PetscFree(vals));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -mat_block_size {{1 2  5 7 8}} -ov {{1 3}} -mat_size {{11 13}} -nd {{7}}

TEST*/
