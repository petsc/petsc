
static char help[] = "Solves a linear system in parallel with MINRES.\n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
  Vec         x, b; /* approx solution, RHS */
  Mat         A;    /* linear system matrix */
  KSP         ksp;  /* linear solver context */
  PC          pc;   /* preconditioner */
  PetscScalar v = 0.0;
  PetscInt    Ii, Istart, Iend, m = 11;
  PetscBool   consistent = PETSC_TRUE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));
  PetscCall(PetscOptionsGetInt(NULL, NULL, "-m", &m, NULL));
  PetscCall(PetscOptionsGetScalar(NULL, NULL, "-vv", &v, NULL));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-consistent", &consistent, NULL));

  /* Create parallel diagonal matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, m, m));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatMPIAIJSetPreallocation(A, 1, NULL, 1, NULL));
  PetscCall(MatSeqAIJSetPreallocation(A, 1, NULL));
  PetscCall(MatSetUp(A));
  PetscCall(MatGetOwnershipRange(A, &Istart, &Iend));

  for (Ii = Istart; Ii < Iend; Ii++) {
    PetscScalar vv = (PetscReal)Ii + 1;
    PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &vv, INSERT_VALUES));
  }

  /* Make A singular or indefinite */
  Ii = m - 1; /* last diagonal entry */
  PetscCall(MatSetValues(A, 1, &Ii, 1, &Ii, &v, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateVecs(A, &x, &b));
  if (consistent) {
    PetscCall(VecSet(x, 1.0));
    PetscCall(MatMult(A, x, b));
    PetscCall(VecSet(x, 0.0));
  } else {
    PetscCall(VecSet(b, 1.0));
  }

  /* Create linear solver context */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetType(ksp, KSPMINRES));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCSetType(pc, PCNONE));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, x));

  /* Test reuse */
  PetscCall(KSPSetInitialGuessNonzero(ksp, PETSC_TRUE));
  PetscCall(VecSet(x, 0.0));
  PetscCall(KSPSolve(ksp, b, x));

  PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

  /* Free work space. */
  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&A));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -ksp_converged_reason

   test:
      suffix: 2
      nsize: 3
      args: -ksp_converged_reason

   test:
      suffix: minres_qlp
      args: -ksp_converged_reason -ksp_minres_qlp -ksp_minres_monitor

   test:
      suffix: minres_qlp_nonconsistent
      args: -ksp_converged_reason -ksp_minres_qlp -ksp_minres_monitor -consistent 0

   test:
      suffix: minres_neg_curve
      args: -ksp_converged_neg_curve -vv -1 -ksp_converged_reason -ksp_minres_qlp {{0 1}}

   test:
      suffix: cg_neg_curve
      args: -ksp_converged_neg_curve -vv -1 -ksp_converged_reason -ksp_type {{cg stcg}}
      requires: !complex

TEST*/
