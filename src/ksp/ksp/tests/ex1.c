
static char help[] = "Tests solving linear system on 0 by 0 matrix, and KSPLSQR convergence test handling.\n\n";

#include <petscksp.h>

static PetscErrorCode GetConvergenceTestName(PetscErrorCode (*converged)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),char name[],size_t n)
{
  PetscFunctionBegin;
  if (converged == KSPConvergedDefault) {
    CHKERRQ(PetscStrncpy(name,"default",n));
  } else if (converged == KSPConvergedSkip) {
    CHKERRQ(PetscStrncpy(name,"skip",n));
  } else if (converged == KSPLSQRConvergedDefault) {
    CHKERRQ(PetscStrncpy(name,"lsqr",n));
  } else {
    CHKERRQ(PetscStrncpy(name,"other",n));
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            C;
  PetscInt       N = 0;
  Vec            u,b,x;
  KSP            ksp;
  PetscReal      norm;
  PetscBool      flg=PETSC_FALSE;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));

  /* create stiffness matrix */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  /* create right hand side and solution */
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&u));
  CHKERRQ(VecSetSizes(u,PETSC_DECIDE,N));
  CHKERRQ(VecSetFromOptions(u));
  CHKERRQ(VecDuplicate(u,&b));
  CHKERRQ(VecDuplicate(u,&x));
  CHKERRQ(VecSet(u,0.0));
  CHKERRQ(VecSet(b,0.0));

  CHKERRQ(VecAssemblyBegin(b));
  CHKERRQ(VecAssemblyEnd(b));

  /* solve linear system */
  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,C,C));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPSolve(ksp,b,u));

  /* test proper handling of convergence test by KSPLSQR */
  CHKERRQ(PetscOptionsGetBool(NULL,NULL,"-test_lsqr",&flg,NULL));
  if (flg) {
    char                  *type;
    char                  convtestname[16];
    PetscBool             islsqr;
    PetscErrorCode        (*converged)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*);
    PetscErrorCode        (*converged1)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*);
    PetscErrorCode        (*destroy)(void*),(*destroy1)(void*);
    void                  *ctx,*ctx1;

    {
      const char *typeP;
      CHKERRQ(KSPGetType(ksp,&typeP));
      CHKERRQ(PetscStrallocpy(typeP,&type));
    }
    CHKERRQ(PetscStrcmp(type,KSPLSQR,&islsqr));
    CHKERRQ(KSPGetConvergenceTest(ksp,&converged,&ctx,&destroy));
    CHKERRQ(GetConvergenceTestName(converged,convtestname,16));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"convergence test: %s\n",convtestname));
    CHKERRQ(KSPSetType(ksp,KSPLSQR));
    CHKERRQ(KSPGetConvergenceTest(ksp,&converged1,&ctx1,&destroy1));
    PetscCheckFalse(converged1 != KSPLSQRConvergedDefault,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test should be KSPLSQRConvergedDefault");
    PetscCheckFalse(destroy1 != KSPConvergedDefaultDestroy,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test destroy function should be KSPConvergedDefaultDestroy");
    if (islsqr) {
      PetscCheckFalse(converged1 != converged,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test should be kept");
      PetscCheckFalse(destroy1 != destroy,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test destroy function should be kept");
      PetscCheckFalse(ctx1 != ctx,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test context should be kept");
    }
    CHKERRQ(GetConvergenceTestName(converged1,convtestname,16));
    CHKERRQ(KSPViewFromOptions(ksp,NULL,"-ksp1_view"));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"convergence test: %s\n",convtestname));
    CHKERRQ(KSPSetType(ksp,type));
    CHKERRQ(KSPGetConvergenceTest(ksp,&converged1,&ctx1,&destroy1));
    PetscCheckFalse(converged1 != converged,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test not reverted properly");
    PetscCheckFalse(destroy1 != destroy,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test destroy function not reverted properly");
    PetscCheckFalse(ctx1 != ctx,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test context not reverted properly");
    CHKERRQ(GetConvergenceTestName(converged1,convtestname,16));
    CHKERRQ(KSPViewFromOptions(ksp,NULL,"-ksp2_view"));
    CHKERRQ(PetscPrintf(PETSC_COMM_WORLD,"convergence test: %s\n",convtestname));
    CHKERRQ(PetscFree(type));
  }

  CHKERRQ(MatMult(C,u,x));
  CHKERRQ(VecAXPY(x,-1.0,b));
  CHKERRQ(VecNorm(x,NORM_2,&norm));

  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(VecDestroy(&u));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
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
