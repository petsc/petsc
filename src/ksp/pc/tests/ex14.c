static char help[] = "Tests that PCApplyTranspose() is the transpose of PCApply() for PCMG.\n\n";

/*
  Checks that PCApplyTranspose() is the transpose of PCApply() for PCMG, for every cycle type,
  with distinct up and down smoothers, and with a restriction that is not the transpose of the
  interpolation (-distinct_restriction). The operator is a nonsymmetric 1D upwind
  advection-diffusion stencil, the hierarchy is built by hand, and the smoothers come from the
  options database. Two checks are made: the bilinear identity y . (M x) == x . (M^T y) on
  random vectors, and M and M^T assembled column by column and compared entry for entry.

  -n must be of the form m * 2^(levels-1) + 1.
*/

#include <petscksp.h>

/*
  Writes the 1D upwind advection-diffusion stencil into A. The advection makes the sub and
  super diagonals differ, i.e., A != A^T, and the shift keeps every row strictly diagonally
  dominant.
*/
static PetscErrorCode SetOperatorValues1D(Mat A, PetscInt n, PetscReal advection, PetscReal shift)
{
  PetscInt row_start, row_end;

  PetscFunctionBeginUser;
  PetscCall(MatGetOwnershipRange(A, &row_start, &row_end));
  for (PetscInt i = row_start; i < row_end; i++) {
    PetscInt    cols[3], n_cols = 0;
    PetscScalar vals[3];

    if (i > 0) {
      cols[n_cols] = i - 1;
      vals[n_cols] = -1.0 - advection;
      n_cols++;
    }
    cols[n_cols] = i;
    vals[n_cols] = 2.0 + advection + shift;
    n_cols++;
    if (i < n - 1) {
      cols[n_cols] = i + 1;
      vals[n_cols] = -1.0;
      n_cols++;
    }
    PetscCall(MatSetValues(A, 1, &i, n_cols, cols, vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode BuildOperator(PetscInt n, PetscReal advection, PetscReal shift, Mat *A_out)
{
  Mat A;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSeqAIJSetPreallocation(A, 3, NULL));
  PetscCall(MatMPIAIJSetPreallocation(A, 3, NULL, 2, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(SetOperatorValues1D(A, n, advection, shift));
  *A_out = A;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Standard 1D linear interpolation from n_coarse points to n_fine = 2 * n_coarse - 1 points -
  fine point 2j takes coarse point j, fine point 2j+1 the average of coarse points j and j+1.
*/
static PetscErrorCode BuildInterpolation(PetscInt n_fine, PetscInt n_coarse, Mat *P_out)
{
  Mat      P;
  PetscInt row_start, row_end;

  PetscFunctionBeginUser;
  PetscCall(MatCreate(PETSC_COMM_WORLD, &P));
  PetscCall(MatSetSizes(P, PETSC_DECIDE, PETSC_DECIDE, n_fine, n_coarse));
  PetscCall(MatSetType(P, MATAIJ));
  PetscCall(MatSeqAIJSetPreallocation(P, 2, NULL));
  PetscCall(MatMPIAIJSetPreallocation(P, 2, NULL, 2, NULL));
  PetscCall(MatGetOwnershipRange(P, &row_start, &row_end));

  for (PetscInt i = row_start; i < row_end; i++) {
    PetscInt    cols[2], n_cols;
    PetscScalar vals[2];

    if (i % 2 == 0) {
      n_cols  = 1;
      cols[0] = i / 2;
      vals[0] = 1.0;
    } else {
      n_cols  = 2;
      cols[0] = (i - 1) / 2;
      cols[1] = (i + 1) / 2;
      vals[0] = 0.5;
      vals[1] = 0.5;
    }
    PetscCall(MatSetValues(P, 1, &i, n_cols, cols, vals, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(P, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(P, MAT_FINAL_ASSEMBLY));

  *P_out = P;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  A restriction that is deliberately not the transpose of the interpolation, R = D P^T with D
  a non-constant diagonal. A transposed cycle that restricted with R^T where it should have
  used P, or interpolated with P where it should have used R^T, is invisible when R = P^T.
*/
static PetscErrorCode BuildRestriction(Mat P, Mat *R_out)
{
  Mat      R;
  Vec      d;
  PetscInt row_start, row_end, n_coarse;

  PetscFunctionBeginUser;
  PetscCall(MatTranspose(P, MAT_INITIAL_MATRIX, &R));
  PetscCall(MatGetSize(R, &n_coarse, NULL));
  /* The left Vec of R, so the diagonal matches the row layout MatDiagonalScale() scales */
  PetscCall(MatCreateVecs(R, NULL, &d));
  PetscCall(VecGetOwnershipRange(d, &row_start, &row_end));
  for (PetscInt j = row_start; j < row_end; j++) {
    PetscScalar value = 1.0 + 0.5 * ((PetscReal)j / (PetscReal)n_coarse);

    PetscCall(VecSetValue(d, j, value, INSERT_VALUES));
  }
  PetscCall(VecAssemblyBegin(d));
  PetscCall(VecAssemblyEnd(d));
  PetscCall(MatDiagonalScale(R, d, NULL));
  PetscCall(VecDestroy(&d));

  *R_out = R;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Builds the multigrid hierarchy by hand, with no DM, so that the interpolation and the
  restriction are exactly what this test wants them to be. The coarse operators are formed by
  PCMG itself as R A P.
*/
static PetscErrorCode BuildHierarchy(PC pc, PetscInt levels, PetscInt n, PetscBool distinct_restriction)
{
  PetscInt *level_size;

  PetscFunctionBeginUser;
  PetscCall(PetscMalloc1(levels, &level_size));
  level_size[levels - 1] = n;
  for (PetscInt l = levels - 2; l >= 0; l--) level_size[l] = (level_size[l + 1] + 1) / 2;

  PetscCall(PCMGSetLevels(pc, levels, NULL));
  PetscCall(PCMGSetGalerkin(pc, PC_MG_GALERKIN_BOTH));

  for (PetscInt l = 1; l < levels; l++) {
    Mat P;

    PetscCall(BuildInterpolation(level_size[l], level_size[l - 1], &P));
    PetscCall(PCMGSetInterpolation(pc, l, P));
    /* With no restriction set PCMG uses P^T */
    if (distinct_restriction) {
      Mat R;

      PetscCall(BuildRestriction(P, &R));
      PetscCall(PCMGSetRestriction(pc, l, R));
      PetscCall(MatDestroy(&R));
    }
    PetscCall(MatDestroy(&P));
  }

  PetscCall(PetscFree(level_size));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The transpose identity itself. Uses VecTDot() rather than VecDot() so that this is the
  bilinear form in both real and complex builds - PCApplyTranspose() is the true transpose,
  not the Hermitian transpose.
*/
static PetscErrorCode CheckTransposeIdentity(PC pc, Mat A, PetscRandom rand, PetscInt n_pairs, PetscReal tol)
{
  Vec x, y, mx, mty;

  PetscFunctionBeginUser;
  PetscCall(MatCreateVecs(A, &x, &mx));
  PetscCall(MatCreateVecs(A, &y, &mty));

  for (PetscInt i = 0; i < n_pairs; i++) {
    PetscScalar lhs, rhs;
    PetscReal   diff, denom;

    /* Random vectors - constant ones would hide a row/column mixup */
    PetscCall(VecSetRandom(x, rand));
    PetscCall(VecSetRandom(y, rand));

    PetscCall(PCApply(pc, x, mx));
    PetscCall(PCApplyTranspose(pc, y, mty));

    PetscCall(VecTDot(mx, y, &lhs));
    PetscCall(VecTDot(x, mty, &rhs));

    diff  = PetscAbsScalar(lhs - rhs);
    denom = PetscMax(PetscAbsScalar(lhs), PetscAbsScalar(rhs));
    if (denom < 1.0) denom = 1.0;

    PetscCheck(diff / denom <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "bilinear identity check: y.(Mx) != x.(M^T y) on pair %" PetscInt_FMT ", relative difference %g > %g", i, (double)(diff / denom), (double)tol);
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&mx));
  PetscCall(VecDestroy(&mty));
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  Builds the preconditioner out explicitly, one column at a time, either as M or as M^T.
*/
static PetscErrorCode BuildExplicit(PC pc, Mat A, PetscInt n, PetscBool transpose, Mat *out)
{
  Mat      dense;
  Vec      e;
  PetscInt local_rows;

  PetscFunctionBeginUser;
  PetscCall(MatGetLocalSize(A, &local_rows, NULL));
  PetscCall(MatCreateDense(PetscObjectComm((PetscObject)A), local_rows, PETSC_DECIDE, n, n, NULL, &dense));
  PetscCall(MatCreateVecs(A, NULL, &e));

  for (PetscInt j = 0; j < n; j++) {
    Vec col;

    PetscCall(VecZeroEntries(e));
    PetscCall(VecSetValue(e, j, 1.0, INSERT_VALUES));
    PetscCall(VecAssemblyBegin(e));
    PetscCall(VecAssemblyEnd(e));

    PetscCall(MatDenseGetColumnVecWrite(dense, j, &col));
    if (transpose) PetscCall(PCApplyTranspose(pc, e, col));
    else PetscCall(PCApply(pc, e, col));
    PetscCall(MatDenseRestoreColumnVecWrite(dense, j, &col));
  }
  PetscCall(VecDestroy(&e));

  *out = dense;
  PetscFunctionReturn(PETSC_SUCCESS);
}

/*
  The sharp version of the check - build M and M^T out column by column and compare every
  entry, rather than the single number the bilinear identity gives.
*/
static PetscErrorCode CheckTransposeExplicitly(PC pc, Mat A, PetscInt n, PetscReal tol)
{
  Mat       m, mt, mt_transposed;
  PetscReal diff, scale;

  PetscFunctionBeginUser;
  PetscCall(BuildExplicit(pc, A, n, PETSC_FALSE, &m));
  PetscCall(BuildExplicit(pc, A, n, PETSC_TRUE, &mt));

  /* (M^T)^T has to be M, entry for entry */
  PetscCall(MatTranspose(mt, MAT_INITIAL_MATRIX, &mt_transposed));
  PetscCall(MatNorm(m, NORM_FROBENIUS, &scale));
  PetscCall(MatAXPY(mt_transposed, -1.0, m, SAME_NONZERO_PATTERN));
  PetscCall(MatNorm(mt_transposed, NORM_FROBENIUS, &diff));

  if (scale < 1.0) scale = 1.0;
  PetscCheck(diff / scale <= tol, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "explicit transpose check: the explicit PCApplyTranspose() matrix is not the transpose of the explicit PCApply() matrix, relative Frobenius difference %g > %g", (double)(diff / scale), (double)tol);

  PetscCall(MatDestroy(&m));
  PetscCall(MatDestroy(&mt));
  PetscCall(MatDestroy(&mt_transposed));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat         A;
  PC          pc;
  PetscRandom rand;
  PetscInt    n = 33, levels = 3, n_pairs = 5, coarsening = 1;
  PetscReal   advection = 1.0, shift = 0.1, tol = PETSC_SMALL;
  PetscBool   distinct_restriction = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n", &n, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-levels", &levels, NULL));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-n_pairs", &n_pairs, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-advection", &advection, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-shift", &shift, NULL));
  PetscCall(PetscOptionsGetReal(NULL, NULL, "-check_tol", &tol, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-distinct_restriction", &distinct_restriction, NULL));

  PetscCheck(levels > 1, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-levels must be at least 2, not %" PetscInt_FMT, levels);
  for (PetscInt l = 0; l < levels - 1; l++) coarsening *= 2;
  PetscCheck(n > 2 * coarsening && (n - 1) % coarsening == 0, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "-n %" PetscInt_FMT " must be larger than %" PetscInt_FMT " and of the form m * 2^(levels-1) + 1 so every level is a valid coarsening of the one above it", n, 2 * coarsening);

  PetscCall(BuildOperator(n, advection, shift, &A));

  /* Seeded so a failure is reproducible */
  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetFromOptions(rand));
  PetscCall(PetscRandomSetSeed(rand, 314159));
  PetscCall(PetscRandomSeed(rand));

  /* The cycle type and the smoothers all come from the options database */
  PetscCall(PCCreate(PETSC_COMM_WORLD, &pc));
  PetscCall(PCSetType(pc, PCMG));
  PetscCall(PCSetOperators(pc, A, A));
  PetscCall(BuildHierarchy(pc, levels, n, distinct_restriction));
  PetscCall(PCSetFromOptions(pc));
  PetscCall(PCSetUp(pc));
  /* The PC is not driven by a KSP, so -pc_view has to be asked for explicitly */
  PetscCall(PCViewFromOptions(pc, NULL, "-pc_view"));

  PetscCall(CheckTransposeIdentity(pc, A, rand, n_pairs, tol));
  PetscCall(CheckTransposeExplicitly(pc, A, n, tol));
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "PCApplyTranspose() is the transpose of PCApply()\n"));

  PetscCall(PCDestroy(&pc));
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   testset:
      args: -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -mg_levels_ksp_max_it 2 -mg_coarse_ksp_type richardson -mg_coarse_pc_type jacobi -mg_coarse_ksp_max_it 4 -mg_coarse_ksp_norm_type none
      output_file: output/ex14_1.out
      nsize: {{1 2}}

      test:
         suffix: cycles
         args: -pc_mg_type {{additive multiplicative full kaskade}shared output} -distinct_restriction {{0 1}shared output}

      test:
         suffix: cycles_distinct_smoothup
         args: -pc_mg_type {{additive multiplicative full kaskade}shared output} -distinct_restriction {{0 1}shared output} -pc_mg_distinct_smoothup -mg_levels_up_ksp_type richardson -mg_levels_up_pc_type jacobi -mg_levels_up_ksp_max_it 1 -mg_levels_up_ksp_richardson_scale 0.7

      test:
         suffix: w_cycle
         args: -pc_mg_type multiplicative -pc_mg_cycle_type w -distinct_restriction {{0 1}shared output}

      test:
         suffix: w_cycle_distinct_smoothup
         args: -pc_mg_type multiplicative -pc_mg_cycle_type w -distinct_restriction {{0 1}shared output} -pc_mg_distinct_smoothup -mg_levels_up_ksp_type richardson -mg_levels_up_pc_type jacobi -mg_levels_up_ksp_max_it 1 -mg_levels_up_ksp_richardson_scale 0.7

      test:
         suffix: multiplicative_cycles
         args: -pc_mg_type multiplicative -pc_mg_multiplicative_cycles 2 -distinct_restriction {{0 1}shared output}

      test:
         suffix: multiplicative_cycles_distinct_smoothup
         args: -pc_mg_type multiplicative -pc_mg_multiplicative_cycles 2 -distinct_restriction {{0 1}shared output} -pc_mg_distinct_smoothup -mg_levels_up_ksp_type richardson -mg_levels_up_pc_type jacobi -mg_levels_up_ksp_max_it 1 -mg_levels_up_ksp_richardson_scale 0.7

   test:
      suffix: lu_coarse
      nsize: 1
      output_file: output/ex14_1.out
      args: -mg_levels_ksp_type richardson -mg_levels_pc_type jacobi -mg_levels_ksp_max_it 2 -mg_coarse_ksp_type preonly -mg_coarse_pc_type lu -pc_mg_type {{additive multiplicative full kaskade}shared output} -distinct_restriction {{0 1}shared output}

TEST*/
