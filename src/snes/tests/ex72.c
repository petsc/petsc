static char help[] = "Tests that SNESSetConvergenceTest() is preserved when SNESSetType() changes to a VI type.\n\n";

#include <petscsnes.h>

typedef struct {
  PetscBool called;
} TestCtx;

static PetscErrorCode CustomConvergedTest(SNES snes, PetscInt it, PetscReal xnorm, PetscReal snorm, PetscReal fnorm, SNESConvergedReason *reason, PetscCtx ctx)
{
  TestCtx *tctx = (TestCtx *)ctx;

  PetscFunctionBeginUser;
  tctx->called = PETSC_TRUE;
  PetscCall(SNESConvergedDefault(snes, it, xnorm, snorm, fnorm, reason, NULL));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecCopy(x, f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

static PetscErrorCode FormJacobian(SNES snes, Vec x, Mat J, Mat B, PetscCtx ctx)
{
  PetscInt    idx = 0;
  PetscScalar one = 1.0;

  PetscFunctionBeginUser;
  PetscCall(MatSetValues(B, 1, &idx, 1, &idx, &one, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  SNES    snes;
  Vec     x, r, xl, xu;
  Mat     J;
  TestCtx ctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  ctx.called = PETSC_FALSE;

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 1));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecSet(x, 0.5));
  PetscCall(VecDuplicate(x, &r));
  PetscCall(VecDuplicate(x, &xl));
  PetscCall(VecDuplicate(x, &xu));
  PetscCall(VecSet(xl, 0.0));
  PetscCall(VecSet(xu, 1.0));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 1, 1));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetFunction(snes, r, FormFunction, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, NULL));

  /* Set custom convergence test BEFORE setting the VI type to verify it is preserved */
  PetscCall(SNESSetConvergenceTest(snes, CustomConvergedTest, &ctx, NULL));

  PetscCall(SNESSetType(snes, SNESVINEWTONRSLS));
  PetscCall(SNESVISetVariableBounds(snes, xl, xu));
  PetscCall(SNESSetFromOptions(snes));

  PetscCall(SNESSolve(snes, NULL, x));

  PetscCheck(ctx.called, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Custom convergence test was not invoked; SNESSetType() overwrote it");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Custom convergence test preserved after SNESSetType() to SNESVINEWTONRSLS\n"));

  PetscCall(SNESDestroy(&snes));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(VecDestroy(&xl));
  PetscCall(VecDestroy(&xu));
  PetscCall(MatDestroy(&J));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 1
      output_file: output/ex72.out

TEST*/
