#include <petscksp.h>

static char help[] = "Checks per-RHS convergence of HPDDM pseudo-block and block Krylov methods.\n";

int main(int argc, char **argv)
{
  Mat                A, B, X;
  KSP                ksp;
  PC                 pc;
  Vec                b, x, r;
  PetscReal          rtol, atol, bnorm, rnorm, reference, tolerance[4];
  PetscInt           start, end, m, n, nrhs, nsolve, its;
  PetscBool          nonzero_guess = PETSC_FALSE, extended = PETSC_FALSE, initial_norm = PETSC_FALSE, min_initial_norm = PETSC_FALSE, check_converged_maxits = PETSC_FALSE, zero_rhs = PETSC_FALSE, exact_guess = PETSC_FALSE;
  KSPConvergedReason reason;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-nonzero_guess", &nonzero_guess, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-extended", &extended, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-check_converged_maxits", &check_converged_maxits, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-zero_rhs", &zero_rhs, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-exact_guess", &exact_guess, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ksp_converged_use_initial_residual_norm", &initial_norm, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-ksp_converged_use_min_initial_residual_norm", &min_initial_norm, NULL));
  n      = extended == PETSC_TRUE ? 6 : 2;
  nrhs   = extended == PETSC_TRUE ? 4 : 2;
  nsolve = zero_rhs == PETSC_TRUE ? 5 : (extended == PETSC_TRUE ? 2 : 1);
  if (extended == PETSC_TRUE) nonzero_guess = PETSC_TRUE;
  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, n, n, 1, NULL, 0, NULL, &A));
  PetscCall(MatGetOwnershipRange(A, &start, &end));
  for (PetscInt i = start; i < end; ++i) PetscCall(MatSetValue(A, i, i, extended == PETSC_FALSE && nonzero_guess == PETSC_TRUE ? 1.0 : i + 1.0, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatGetLocalSize(A, &m, NULL));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, n, nrhs, NULL, &B));
  PetscCall(MatCreateDense(PETSC_COMM_WORLD, m, PETSC_DECIDE, n, nrhs, NULL, &X));
  PetscCall(MatCreateVecs(A, NULL, &r));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetType(ksp, KSPHPDDM));
  PetscCall(KSPHPDDMSetType(ksp, KSP_HPDDM_TYPE_GMRES));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetTolerances(ksp, extended == PETSC_TRUE ? 1.e-8 : 1.e-4, 1.e-12, PETSC_CURRENT, extended == PETSC_TRUE ? 100 : 10));
  PetscCall(KSPSetInitialGuessNonzero(ksp, (PetscBool)(nonzero_guess == PETSC_TRUE || zero_rhs == PETSC_TRUE || exact_guess == PETSC_TRUE)));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetTolerances(ksp, &rtol, &atol, NULL, NULL));

  for (PetscInt solve = 0; solve < nsolve; ++solve) {
    PetscCall(MatZeroEntries(B));
    PetscCall(MatZeroEntries(X));
    if (zero_rhs == PETSC_TRUE) {
      // move the zero column, then solve zero RHS from zero and nonzero initial guesses
      for (PetscInt j = 0; j < nrhs; ++j) {
        PetscBool zero = (PetscBool)(solve == 2 || solve == 4 || j == (solve == 1 ? 1 : 0));

        for (PetscInt i = start; i < end; ++i) {
          if (zero == PETSC_FALSE) PetscCall(MatSetValue(B, i, j, i + 1.0, INSERT_VALUES));
          if (zero == PETSC_TRUE && solve >= 3) PetscCall(MatSetValue(X, i, j, 1.0, INSERT_VALUES));
        }
      }
    } else if (extended == PETSC_TRUE) {
      // reverse the RHS scales on the second solve to exercise reused convergence and recycling state
      for (PetscInt j = 0; j < 2; ++j) {
        PetscReal scale = j == solve ? 10000.0 : 1.0, factor = j == 0 ? 10.0 : 0.01;

        for (PetscInt i = start; i < end; ++i) {
          PetscCall(MatSetValue(B, i, j, scale * (i + 1.0), INSERT_VALUES));
          PetscCall(MatSetValue(X, i, j, scale * (1.0 - factor), INSERT_VALUES));
        }
      }
      // RHS #2 is 0, and RHS #3 starts with an exact solution
      for (PetscInt i = start; i < end; ++i) {
        PetscCall(MatSetValue(B, i, 3, i == 0 ? 1.0 : 0.0, INSERT_VALUES));
        PetscCall(MatSetValue(X, i, 3, i == 0 ? 1.0 : 0.0, INSERT_VALUES));
      }
    } else if (nonzero_guess == PETSC_TRUE) {
      // each initial residual exceeds its own tolerance but is below rtol times the Frobenius norm of B
      for (PetscInt i = start; i < end; ++i) {
        PetscCall(MatSetValue(B, i, i, 1.0, INSERT_VALUES));
        PetscCall(MatSetValue(X, i, i, 1.0 - 1.2e-4, INSERT_VALUES));
      }
    } else {
      // the first RHS is solved in one iteration; the smaller second RHS needs another
      for (PetscInt i = start; i < end; ++i) {
        PetscCall(MatSetValue(B, i, 0, i == 0 ? 10000.0 : 0.0, INSERT_VALUES));
        PetscCall(MatSetValue(B, i, 1, i == 0 ? 1.0 : 2.0, INSERT_VALUES));
      }
    }
    PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyBegin(X, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(X, MAT_FINAL_ASSEMBLY));
    if (exact_guess == PETSC_TRUE) {
      PetscCall(MatGetDiagonal(A, r));
      for (PetscInt j = 0; j < nrhs; ++j) {
        PetscCall(MatDenseGetColumnVecRead(B, j, &b));
        PetscCall(MatDenseGetColumnVecWrite(X, j, &x));
        PetscCall(VecPointwiseDivide(x, b, r));
        PetscCall(MatDenseRestoreColumnVecWrite(X, j, &x));
        PetscCall(MatDenseRestoreColumnVecRead(B, j, &b));
      }
    }
    for (PetscInt j = 0; j < nrhs; ++j) {
      PetscCall(MatDenseGetColumnVecRead(B, j, &b));
      PetscCall(MatDenseGetColumnVecRead(X, j, &x));
      PetscCall(MatMult(A, x, r));
      PetscCall(VecAYPX(r, -1.0, b));
      PetscCall(VecNorm(b, NORM_2, &bnorm));
      PetscCall(VecNorm(r, NORM_2, &rnorm));
      reference = bnorm;
      if (initial_norm == PETSC_TRUE || bnorm == 0.0) reference = rnorm;
      else if (min_initial_norm == PETSC_TRUE) reference = PetscMin(bnorm, rnorm);
      tolerance[j] = PetscMax(atol, rtol * reference);
      PetscCall(MatDenseRestoreColumnVecRead(X, j, &x));
      PetscCall(MatDenseRestoreColumnVecRead(B, j, &b));
    }
    PetscCall(KSPMatSolve(ksp, B, X));
    PetscCall(KSPGetConvergedReason(ksp, &reason));
    PetscCheck(reason > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solve #%" PetscInt_FMT " did not converge: %s", solve, KSPConvergedReasons[reason]);
    PetscCheck(check_converged_maxits == PETSC_FALSE || reason == KSP_CONVERGED_ITS, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Expected convergence at the iteration limit, got %s", KSPConvergedReasons[reason]);
    if (exact_guess == PETSC_TRUE || (zero_rhs == PETSC_TRUE && solve == 2)) {
      PetscCall(KSPGetIterationNumber(ksp, &its));
      PetscCheck(its == 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solve #%" PetscInt_FMT " took %" PetscInt_FMT " iterations with an exact initial solution", solve, its);
    }
    for (PetscInt j = 0; j < nrhs; ++j) {
      PetscCall(MatDenseGetColumnVecRead(B, j, &b));
      PetscCall(MatDenseGetColumnVecRead(X, j, &x));
      PetscCall(MatMult(A, x, r));
      PetscCall(VecAYPX(r, -1.0, b));
      PetscCall(VecNorm(r, NORM_2, &rnorm));
      PetscCheck(rnorm <= 1.01 * tolerance[j], PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Solve #%" PetscInt_FMT " RHS #%" PetscInt_FMT " has residual %g above its tolerance %g", solve, j, (double)rnorm, (double)tolerance[j]);
      PetscCall(MatDenseRestoreColumnVecRead(X, j, &x));
      PetscCall(MatDenseRestoreColumnVecRead(B, j, &b));
    }
  }
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&X));
  PetscCall(MatDestroy(&B));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

  build:
    requires: hpddm double

  test:
    nsize: 2
    args: -ksp_hpddm_variant {{left right flexible}shared output}
    output_file: output/empty.out

  test:
    suffix: converged_maxits
    requires: !complex !defined(PETSC_USE_64BIT_INDICES)
    nsize: 2
    args: -ksp_converged_maxits -ksp_max_it 2 -check_converged_maxits
    output_file: output/empty.out

  test:
    suffix: nonzero_guess
    requires: !complex !defined(PETSC_USE_64BIT_INDICES)
    nsize: 2
    args: -nonzero_guess -ksp_hpddm_variant {{left right flexible}shared output}
    output_file: output/empty.out

  testset:
    suffix: nonzero_guess_block
    requires: !complex !defined(PETSC_USE_64BIT_INDICES)
    nsize: 2
    args: -nonzero_guess -ksp_gmres_restart 4
    output_file: output/empty.out
    test:
      suffix: gmres
      args: -ksp_hpddm_type bgmres
    test:
      suffix: gcrodr
      args: -ksp_hpddm_type bgcrodr -ksp_hpddm_recycle 1

  test:
    suffix: cg
    nsize: 2
    args: -ksp_hpddm_type cg -ksp_hpddm_variant {{left flexible}shared output}
    output_file: output/empty.out

  test:
    suffix: gcrodr
    nsize: 2
    args: -ksp_hpddm_type gcrodr -ksp_hpddm_variant right -ksp_gmres_restart 3 -ksp_hpddm_recycle 1
    output_file: output/empty.out

  test:
    suffix: extended
    nsize: 2
    args: -extended -ksp_gmres_restart 3 -ksp_matsolve_batch_size {{4 3}shared output}
    output_file: output/empty.out

  testset:
    suffix: cg_extended
    nsize: 2
    args: -extended -ksp_hpddm_type cg -ksp_hpddm_variant {{left flexible}shared output}
    output_file: output/empty.out
    test:
      suffix: uir
      args: -ksp_converged_use_initial_residual_norm
    test:
      suffix: umir
      args: -ksp_converged_use_min_initial_residual_norm

  testset:
    suffix: gcrodr_extended
    nsize: 2
    args: -extended -ksp_hpddm_type gcrodr -ksp_hpddm_variant {{right flexible}shared output} -ksp_gmres_restart 4 -ksp_hpddm_recycle 2 -ksp_matsolve_batch_size 3
    output_file: output/empty.out
    test:
      suffix: uir
      args: -ksp_converged_use_initial_residual_norm
    test:
      suffix: umir
      args: -ksp_converged_use_min_initial_residual_norm

  testset:
    suffix: gcrodr_extended
    nsize: 2
    args: -extended -ksp_hpddm_type gcrodr -ksp_hpddm_recycle 2
    output_file: output/empty.out
    test:
      suffix: umir_full_batch
      args: -ksp_matsolve_batch_size 4 -ksp_gmres_restart 4 -ksp_hpddm_variant right -ksp_converged_use_min_initial_residual_norm
    test:
      suffix: large_restart
      requires: !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -ksp_matsolve_batch_size 3 -ksp_gmres_restart 30
    test:
      suffix: mixed_precision
      args: -ksp_matsolve_batch_size 3 -ksp_gmres_restart 4 -ksp_hpddm_variant {{right flexible}shared output} -ksp_hpddm_precision single -ksp_rtol 1e-5
    test:
      suffix: gcrodr_extended_strategy_b
      requires: !complex !defined(PETSC_USE_64BIT_INDICES)
      args: -ksp_hpddm_variant flexible -ksp_gmres_restart 4 -ksp_hpddm_recycle_strategy B -ksp_converged_use_min_initial_residual_norm

  testset:
    suffix: zero_rhs
    nsize: 2
    args: -zero_rhs
    output_file: output/empty.out
    test:
      suffix: pseudo
      args: -ksp_hpddm_type {{gmres cg}shared output}
    test:
      suffix: recycle
      args: -ksp_hpddm_type gcrodr -ksp_gmres_restart 4 -ksp_hpddm_recycle 1
    test:
      suffix: block
      args: -ksp_hpddm_type {{bgmres bfbcg}shared output} -ksp_hpddm_deflation_tol 1e-12
    test:
      suffix: block_recycle
      args: -ksp_hpddm_type bgcrodr -ksp_gmres_restart 4 -ksp_hpddm_recycle 1 -ksp_hpddm_deflation_tol 1e-12

  testset:
    suffix: exact_guess
    requires: !complex !defined(PETSC_USE_64BIT_INDICES)
    nsize: 2
    args: -exact_guess
    output_file: output/empty.out
    test:
      suffix: zero_iterations
      args: -ksp_hpddm_type {{gmres cg gcrodr}shared output} -ksp_max_it 0

TEST*/
