static char help[] = "Tests MatCreateKAIJAB() with a general second operand: K = (A x T) + (B x S).\n\n";

#include <petscmat.h>
#include <petsc/private/petscimpl.h> // PetscObjectStateGet()

// Check K2 = (A \otimes T) + (B \otimes S) against the sum of the two single-operand KAIJ matrices
static PetscErrorCode CheckMult(Mat K2, Mat KAT, Mat KBS, Vec x, Vec y2, Vec yr, Vec ytmp, const char label[])
{
  PetscReal nrm, ynrm, tol = 1.e3 * PETSC_MACHINE_EPSILON;

  PetscFunctionBeginUser;
  for (PetscInt i = 0; i < 10; i++) {
    PetscCall(VecSetRandom(x, NULL));
    PetscCall(MatMult(K2, x, y2));
    PetscCall(MatMult(KAT, x, yr));
    PetscCall(MatMult(KBS, x, ytmp));
    PetscCall(VecAXPY(yr, 1.0, ytmp)); // yr = (A x T)x + (B x S)x
    PetscCall(VecNorm(y2, NORM_2, &ynrm));
    PetscCall(VecAXPY(yr, -1.0, y2));
    PetscCall(VecNorm(yr, NORM_2, &nrm));
    PetscCheck(nrm <= tol * (ynrm + 1.0), PETSC_COMM_WORLD, PETSC_ERR_CONV_FAILED, "MatCreateKAIJAB() MatMult mismatch (%s): ||K2*x - ((AxT)+(BxS))*x|| = %g", label, (double)nrm);
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

// With B = beta I, the block diagonal must match that of the single-operand KAIJ with S replaced by beta S
static PetscErrorCode CheckInvertBlockDiagonal(Mat A, PetscInt p, PetscInt q, const PetscScalar S[], const PetscScalar T[])
{
  Mat                Bid, Kab, Kref;
  Vec                d;
  PetscScalar       *Sbeta;
  const PetscScalar  beta = 0.5; // exactly representable, so both paths form beta*S identically
  const PetscScalar *dab, *dref;
  PetscReal          tol = 1.e3 * PETSC_MACHINE_EPSILON;
  PetscInt           nblk, rstart, rend;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  // Bid has A's nonzero structure (hence A's ghost columns) but is numerically beta*I
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &Bid));
  PetscCall(MatZeroEntries(Bid));
  PetscCall(MatCreateVecs(Bid, NULL, &d));
  PetscCall(VecSet(d, beta));
  PetscCall(MatDiagonalSet(Bid, d, INSERT_VALUES));
  PetscCall(VecDestroy(&d));

  PetscCall(PetscMalloc1(p * q, &Sbeta));
  for (PetscInt k = 0; k < p * q; k++) Sbeta[k] = beta * S[k];

  PetscCall(MatCreateKAIJAB(A, Bid, p, q, S, T, &Kab));
  PetscCall(MatCreateKAIJ(A, p, q, Sbeta, T, &Kref));
  PetscCall(MatInvertBlockDiagonal(Kab, &dab));
  PetscCall(MatInvertBlockDiagonal(Kref, &dref));
  nblk = p * p * (rend - rstart);
  for (PetscInt k = 0; k < nblk; k++)
    PetscCheck(PetscAbsScalar(dab[k] - dref[k]) <= tol * (1.0 + PetscAbsScalar(dref[k])), PETSC_COMM_SELF, PETSC_ERR_CONV_FAILED, "MatInvertBlockDiagonal() with a general B differs from the equivalent single-operand KAIJ at entry %" PetscInt_FMT, k);

  PetscCall(PetscFree(Sbeta));
  PetscCall(MatDestroy(&Kab));
  PetscCall(MatDestroy(&Kref));
  PetscCall(MatDestroy(&Bid));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  Mat                A, B, K2, KAT, KBS, Bget, Dblk;
  Vec                x, y2, yr, ytmp;
  PetscScalar       *S, *T;
  PetscScalar        vals[3];
  const PetscScalar *dblk1, *dblk2;
  PetscObjectState   st0, st1, nz0, nz1;
  PetscInt           n = 30, nc, p = 3, q = 3, rstart, rend, cols[3], mk, nk, md, nd, row0 = 0;
  PetscBool          bnull = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-p", &p, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-q", &q, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-b_null", &bnull, NULL));

  // Build a parallel tridiagonal AIJ matrix A (nonzero off-diagonal coupling exercises the MPI ghost path)
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 1, NULL));
  PetscCall(MatGetOwnershipRange(A, &rstart, &rend));
  for (PetscInt i = rstart; i < rend; i++) {
    nc = 0;
    if (i > 0) {
      cols[nc]   = i - 1;
      vals[nc++] = -1.0;
    }
    cols[nc]   = i;
    vals[nc++] = 2.0 + (PetscScalar)i;
    if (i < n - 1) {
      cols[nc]   = i + 1;
      vals[nc++] = -1.0 - 0.5 * (PetscScalar)i;
    }
    PetscCall(MatSetValues(A, 1, &i, nc, cols, vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  // B shares A's nonzero structure but has distinct values
  PetscCall(MatDuplicate(A, MAT_COPY_VALUES, &B));
  PetscCall(MatScale(B, 0.37));

  // Nonsymmetric, diagonally dominant S and T (column-major), so that every block a T + b S is invertible
  PetscCall(PetscMalloc2(p * q, &S, p * q, &T));
  for (PetscInt i = 0; i < p; i++) {
    for (PetscInt j = 0; j < q; j++) {
      S[i + p * j] = (i == j ? 2.0 : 0.0) + ((PetscReal)((i + 1) * (j + 2))) / ((PetscReal)(p + q));
      T[i + p * j] = (i == j ? 2.0 : 0.0) + ((PetscReal)((p - i) + 2 * j + 1)) / ((PetscReal)(p * q));
    }
  }

  // K2 = (A \otimes T) + (B \otimes S), or (A \otimes T) + (I \otimes S) with -b_null
  PetscCall(MatCreateKAIJAB(A, bnull ? NULL : B, p, q, S, T, &K2));
  PetscCall(MatKAIJGetB(K2, &Bget));
  PetscCheck(Bget == (bnull ? NULL : B), PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatKAIJGetB() did not return the matrix given to MatCreateKAIJAB()");

  // MatGetDiagonalBlock() must work before the first MatMult()
  PetscCall(MatGetDiagonalBlock(K2, &Dblk));
  PetscCall(MatGetLocalSize(K2, &mk, &nk));
  PetscCall(MatGetLocalSize(Dblk, &md, &nd));
  PetscCheck(md == mk && nd == nk, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatGetDiagonalBlock() returned a %" PetscInt_FMT " x %" PetscInt_FMT " block, expected %" PetscInt_FMT " x %" PetscInt_FMT, md, nd, mk, nk);

  // Reference operators
  PetscCall(MatCreateKAIJ(A, p, q, NULL, T, &KAT));            // A \otimes T
  if (bnull) PetscCall(MatCreateKAIJ(B, p, q, S, NULL, &KBS)); // I \otimes S (matches identity B)
  else PetscCall(MatCreateKAIJ(B, p, q, NULL, S, &KBS));       // B \otimes S

  PetscCall(MatCreateVecs(K2, &x, &y2));
  PetscCall(VecDuplicate(y2, &yr));
  PetscCall(VecDuplicate(y2, &ytmp));

  PetscCall(CheckMult(K2, KAT, KBS, x, y2, yr, ytmp, "initial"));

  // Reassembling an operand must rebuild the cached submatrices of a MATMPIKAIJ
  PetscCall(MatScale(A, 1.3));
  PetscCall(CheckMult(K2, KAT, KBS, x, y2, yr, ytmp, "after MatScale(A)"));
  PetscCall(MatScale(B, -0.8));
  PetscCall(CheckMult(K2, KAT, KBS, x, y2, yr, ytmp, "after MatScale(B)"));

  // The reassembly replaced the diagonal block, so fetch it again
  PetscCall(MatGetDiagonalBlock(K2, &Dblk));
  PetscCall(MatGetLocalSize(Dblk, &md, &nd));
  PetscCheck(md == mk && nd == nk, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatGetDiagonalBlock() returned a %" PetscInt_FMT " x %" PetscInt_FMT " block after reassembly, expected %" PetscInt_FMT " x %" PetscInt_FMT, md, nd, mk, nk);

  // Changing S must invalidate the cached submatrices, which hold a copy of S
  for (PetscInt i = 0; i < p * q; i++) S[i] *= 1.7;
  PetscCall(MatKAIJSetS(K2, p, q, S));
  PetscCall(MatDestroy(&KBS));
  if (bnull) PetscCall(MatCreateKAIJ(B, p, q, S, NULL, &KBS));
  else PetscCall(MatCreateKAIJ(B, p, q, NULL, S, &KBS));
  PetscCall(CheckMult(K2, KAT, KBS, x, y2, yr, ytmp, "after MatKAIJSetS"));

  if (p == q) PetscCall(CheckInvertBlockDiagonal(A, p, q, S, T));

  // The array from MatInvertBlockDiagonal() must survive a rebuild of the cached submatrices
  if (p == q) {
    PetscCall(MatInvertBlockDiagonal(K2, &dblk1));
    // Keep the old submatrix alive, so that a freed buffer cannot be reused at the same address
    PetscCall(MatGetDiagonalBlock(K2, &Dblk));
    PetscCall(PetscObjectReference((PetscObject)Dblk));
    PetscCall(MatScale(A, 1.1));
    PetscCall(MatMult(K2, x, y2)); // rebuilds the cached submatrices, releasing the previous ones
    PetscCall(MatInvertBlockDiagonal(K2, &dblk2));
    PetscCheck(dblk1 == dblk2, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatInvertBlockDiagonal() returned an array that a later rebuild of the cached submatrices invalidated");
    PetscCall(MatDestroy(&Dblk));
    PetscCall(CheckMult(K2, KAT, KBS, x, y2, yr, ytmp, "after MatScale(A) with a cached block diagonal"));
  }

  // Replacing B with the identity after setup must take effect
  PetscCall(MatKAIJSetB(K2, NULL));
  PetscCall(MatKAIJGetB(K2, &Bget));
  PetscCheck(!Bget, PETSC_COMM_SELF, PETSC_ERR_PLIB, "MatKAIJGetB() did not report the identity after MatKAIJSetB(K, NULL)");
  PetscCall(MatDestroy(&KBS));
  PetscCall(MatCreateKAIJ(A, p, q, S, NULL, &KBS)); // I \otimes S
  PetscCall(CheckMult(K2, KAT, KBS, x, y2, yr, ytmp, "after MatKAIJSetB(K, NULL)"));

  // ... and setting it back to a general operand must take effect too
  PetscCall(MatKAIJSetB(K2, B));
  PetscCall(MatDestroy(&KBS));
  PetscCall(MatCreateKAIJ(B, p, q, NULL, S, &KBS)); // B \otimes S
  PetscCall(CheckMult(K2, KAT, KBS, x, y2, yr, ytmp, "after MatKAIJSetB(K, B)"));

  // Assembling K2 after a value-only change to A must raise its state but not its nonzero state
  PetscCall(PetscObjectStateGet((PetscObject)K2, &st0));
  PetscCall(MatGetNonzeroState(K2, &nz0));
  PetscCall(MatScale(A, 2.0)); // value change, same nonzero pattern
  PetscCall(MatAssemblyBegin(K2, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(K2, MAT_FINAL_ASSEMBLY));
  PetscCall(PetscObjectStateGet((PetscObject)K2, &st1));
  PetscCall(MatGetNonzeroState(K2, &nz1));
  PetscCheck(st1 != st0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Assembling the KAIJ matrix after reassembling an operand did not raise its object state");
  PetscCheck(nz1 == nz0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A value-only change to an operand was reported as a nonzero-pattern change of the KAIJ matrix");

  // A new nonzero in A must raise the nonzero state of K2, and row 0 has a free preallocated slot for (0, 2)
  if (n > 2) {
    PetscCall(MatSetOption(A, MAT_NEW_NONZERO_ALLOCATION_ERR, PETSC_FALSE));
    if (rstart == 0) {
      cols[0] = 2;
      vals[0] = 0.25;
      PetscCall(MatSetValues(A, 1, &row0, 1, cols, vals, INSERT_VALUES));
    }
    PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
    PetscCall(MatGetNonzeroState(K2, &nz0));
    PetscCall(MatAssemblyBegin(K2, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(K2, MAT_FINAL_ASSEMBLY));
    PetscCall(MatGetNonzeroState(K2, &nz1));
    PetscCheck(nz1 != nz0, PETSC_COMM_SELF, PETSC_ERR_PLIB, "A new nonzero in an operand was not forwarded to the KAIJ matrix's nonzero state");
  }

  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "MatCreateKAIJAB() MatMult matches (A x T) + (B x S)\n"));

  PetscCall(PetscFree2(S, T));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y2));
  PetscCall(VecDestroy(&yr));
  PetscCall(VecDestroy(&ytmp));
  PetscCall(MatDestroy(&K2));
  PetscCall(MatDestroy(&KAT));
  PetscCall(MatDestroy(&KBS));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&B));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  test:
    suffix: 1
    nsize: {{1 2 4}}
    args: -p {{2 3}} -q {{2 3}}

  test:
    suffix: b_null
    nsize: {{1 3}}
    args: -b_null -p 3 -q 3

TEST*/
