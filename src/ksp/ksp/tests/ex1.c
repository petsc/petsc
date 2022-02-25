
static char help[] = "Tests solving linear system on 0 by 0 matrix, and KSPLSQR convergence test handling.\n\n";

#include <petscksp.h>

static PetscErrorCode GetConvergenceTestName(PetscErrorCode (*converged)(KSP,PetscInt,PetscReal,KSPConvergedReason*,void*),char name[],size_t n)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (converged == KSPConvergedDefault) {
    ierr = PetscStrncpy(name,"default",n);CHKERRQ(ierr);
  } else if (converged == KSPConvergedSkip) {
    ierr = PetscStrncpy(name,"skip",n);CHKERRQ(ierr);
  } else if (converged == KSPLSQRConvergedDefault) {
    ierr = PetscStrncpy(name,"lsqr",n);CHKERRQ(ierr);
  } else {
    ierr = PetscStrncpy(name,"other",n);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc,char **args)
{
  Mat            C;
  PetscErrorCode ierr;
  PetscInt       N = 0;
  Vec            u,b,x;
  KSP            ksp;
  PetscReal      norm;
  PetscBool      flg=PETSC_FALSE;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;

  /* create stiffness matrix */
  ierr = MatCreate(PETSC_COMM_WORLD,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,N,N);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  /* create right hand side and solution */
  ierr = VecCreate(PETSC_COMM_WORLD,&u);CHKERRQ(ierr);
  ierr = VecSetSizes(u,PETSC_DECIDE,N);CHKERRQ(ierr);
  ierr = VecSetFromOptions(u);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(u,&x);CHKERRQ(ierr);
  ierr = VecSet(u,0.0);CHKERRQ(ierr);
  ierr = VecSet(b,0.0);CHKERRQ(ierr);

  ierr = VecAssemblyBegin(b);CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b);CHKERRQ(ierr);

  /* solve linear system */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,C,C);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPSolve(ksp,b,u);CHKERRQ(ierr);

  /* test proper handling of convergence test by KSPLSQR */
  ierr = PetscOptionsGetBool(NULL,NULL,"-test_lsqr",&flg,NULL);CHKERRQ(ierr);
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
      ierr = KSPGetType(ksp,&typeP);CHKERRQ(ierr);
      ierr = PetscStrallocpy(typeP,&type);CHKERRQ(ierr);
    }
    ierr = PetscStrcmp(type,KSPLSQR,&islsqr);CHKERRQ(ierr);
    ierr = KSPGetConvergenceTest(ksp,&converged,&ctx,&destroy);CHKERRQ(ierr);
    ierr = GetConvergenceTestName(converged,convtestname,16);CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"convergence test: %s\n",convtestname);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,KSPLSQR);CHKERRQ(ierr);
    ierr = KSPGetConvergenceTest(ksp,&converged1,&ctx1,&destroy1);CHKERRQ(ierr);
    PetscCheckFalse(converged1 != KSPLSQRConvergedDefault,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test should be KSPLSQRConvergedDefault");
    PetscCheckFalse(destroy1 != KSPConvergedDefaultDestroy,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test destroy function should be KSPConvergedDefaultDestroy");
    if (islsqr) {
      PetscCheckFalse(converged1 != converged,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test should be kept");
      PetscCheckFalse(destroy1 != destroy,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test destroy function should be kept");
      PetscCheckFalse(ctx1 != ctx,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test context should be kept");
    }
    ierr = GetConvergenceTestName(converged1,convtestname,16);CHKERRQ(ierr);
    ierr = KSPViewFromOptions(ksp,NULL,"-ksp1_view");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"convergence test: %s\n",convtestname);CHKERRQ(ierr);
    ierr = KSPSetType(ksp,type);CHKERRQ(ierr);
    ierr = KSPGetConvergenceTest(ksp,&converged1,&ctx1,&destroy1);CHKERRQ(ierr);
    PetscCheckFalse(converged1 != converged,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test not reverted properly");
    PetscCheckFalse(destroy1 != destroy,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test destroy function not reverted properly");
    PetscCheckFalse(ctx1 != ctx,PETSC_COMM_WORLD,PETSC_ERR_PLIB,"convergence test context not reverted properly");
    ierr = GetConvergenceTestName(converged1,convtestname,16);CHKERRQ(ierr);
    ierr = KSPViewFromOptions(ksp,NULL,"-ksp2_view");CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD,"convergence test: %s\n",convtestname);CHKERRQ(ierr);
    ierr = PetscFree(type);CHKERRQ(ierr);
  }

  ierr = MatMult(C,u,x);CHKERRQ(ierr);
  ierr = VecAXPY(x,-1.0,b);CHKERRQ(ierr);
  ierr = VecNorm(x,NORM_2,&norm);CHKERRQ(ierr);

  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&u);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
