
static char help[] = "Tests solving linear system on 0 by 0 matrix, and KSPLSQR convergence test handling.\n\n";

#include <petscksp.h>

static PetscErrorCode GetConvergenceTestName(PetscErrorCode (*converged)(KSP, PetscInt, PetscReal, KSPConvergedReason *, void *), char name[], size_t n)
{
  PetscFunctionBegin;
  if (converged == KSPConvergedDefault) {
    PetscCall(PetscStrncpy(name, "default", n));
  } else if (converged == KSPConvergedSkip) {
    PetscCall(PetscStrncpy(name, "skip", n));
  } else if (converged == KSPLSQRConvergedDefault) {
    PetscCall(PetscStrncpy(name, "lsqr", n));
  } else {
    PetscCall(PetscStrncpy(name, "other", n));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **args)
{
  Mat       C;
  PetscInt  N = 0;
  Vec       u, b, x;
  KSP       ksp;
  PetscReal norm;
  PetscBool flg = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, (char *)0, help));

  /* create stiffness matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &C));
  PetscCall(MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, N, N));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));
  PetscCall(MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY));

  /* create right hand side and solution */
  PetscCall(VecCreate(PETSC_COMM_WORLD, &u));
  PetscCall(VecSetSizes(u, PETSC_DECIDE, N));
  PetscCall(VecSetFromOptions(u));
  PetscCall(VecDuplicate(u, &b));
  PetscCall(VecDuplicate(u, &x));
  PetscCall(VecSet(u, 0.0));
  PetscCall(VecSet(b, 0.0));

  PetscCall(VecAssemblyBegin(b));
  PetscCall(VecAssemblyEnd(b));

  /* solve linear system */
  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, C, C));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPSolve(ksp, b, u));

  /* test proper handling of convergence test by KSPLSQR */
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_lsqr", &flg, NULL));
  if (flg) {
    char     *type;
    char      convtestname[16];
    PetscBool islsqr;
    PetscErrorCode (*converged)(KSP, PetscInt, PetscReal, KSPConvergedReason *, void *);
    PetscErrorCode (*converged1)(KSP, PetscInt, PetscReal, KSPConvergedReason *, void *);
    PetscErrorCode (*destroy)(void *), (*destroy1)(void *);
    void *ctx, *ctx1;

    {
      const char *typeP;
      PetscCall(KSPGetType(ksp, &typeP));
      PetscCall(PetscStrallocpy(typeP, &type));
    }
    PetscCall(PetscStrcmp(type, KSPLSQR, &islsqr));
    PetscCall(KSPGetConvergenceTest(ksp, &converged, &ctx, &destroy));
    PetscCall(GetConvergenceTestName(converged, convtestname, 16));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "convergence test: %s\n", convtestname));
    PetscCall(KSPSetType(ksp, KSPLSQR));
    PetscCall(KSPGetConvergenceTest(ksp, &converged1, &ctx1, &destroy1));
    PetscCheck(converged1 == KSPLSQRConvergedDefault, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "convergence test should be KSPLSQRConvergedDefault");
    PetscCheck(destroy1 == KSPConvergedDefaultDestroy, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "convergence test destroy function should be KSPConvergedDefaultDestroy");
    if (islsqr) {
      PetscCheck(converged1 == converged, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "convergence test should be kept");
      PetscCheck(destroy1 == destroy, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "convergence test destroy function should be kept");
      PetscCheck(ctx1 == ctx, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "convergence test context should be kept");
    }
    PetscCall(GetConvergenceTestName(converged1, convtestname, 16));
    PetscCall(KSPViewFromOptions(ksp, NULL, "-ksp1_view"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "convergence test: %s\n", convtestname));
    PetscCall(KSPSetType(ksp, type));
    PetscCall(KSPGetConvergenceTest(ksp, &converged1, &ctx1, &destroy1));
    PetscCheck(converged1 == converged, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "convergence test not reverted properly");
    PetscCheck(destroy1 == destroy, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "convergence test destroy function not reverted properly");
    PetscCheck(ctx1 == ctx, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "convergence test context not reverted properly");
    PetscCall(GetConvergenceTestName(converged1, convtestname, 16));
    PetscCall(KSPViewFromOptions(ksp, NULL, "-ksp2_view"));
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "convergence test: %s\n", convtestname));
    PetscCall(PetscFree(type));
  }

  PetscCall(MatMult(C, u, x));
  PetscCall(VecAXPY(x, -1.0, b));
  PetscCall(VecNorm(x, NORM_2, &norm));

  PetscCall(KSPDestroy(&ksp));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args:  -pc_type jacobi -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

    test:
      suffix: 2
      nsize: 2
      args: -pc_type jacobi -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

    test:
      suffix: 3
      args: -pc_type sor -pc_sor_symmetric -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

    test:
      suffix: 5
      args: -pc_type eisenstat -ksp_monitor_short -ksp_gmres_cgs_refinement_type refine_always

    testset:
      args: -test_lsqr -ksp{,1,2}_view -pc_type jacobi
      filter: grep -E "(^  type:|preconditioning|norm type|convergence test:)"
      test:
        suffix: lsqr_0
        args: -ksp_convergence_test {{default skip}separate output}
      test:
        suffix: lsqr_1
        args: -ksp_type cg -ksp_convergence_test {{default skip}separate output}
      test:
        suffix: lsqr_2
        args: -ksp_type lsqr

TEST*/
