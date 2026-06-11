static char help[] = "Tests that SNESComputeJacobian() calls the user Jacobian function when a left NPC is active.\n\n";

#include <petscsnes.h>

typedef struct {
  PetscInt jac_calls;
} AppCtx;

PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, PetscCtx ctx)
{
  PetscFunctionBeginUser;
  PetscCall(VecZeroEntries(f));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat B, PetscCtx ctx)
{
  AppCtx *appctx = (AppCtx *)ctx;

  PetscFunctionBeginUser;
  appctx->jac_calls++;
  PetscCall(MatAssemblyBegin(B, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(B, MAT_FINAL_ASSEMBLY));
  if (jac != B) {
    PetscCall(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv)
{
  SNES   snes, npc;
  Vec    x, r;
  Mat    J;
  AppCtx appctx;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));

  appctx.jac_calls = 0;

  PetscCall(VecCreate(PETSC_COMM_WORLD, &x));
  PetscCall(VecSetSizes(x, PETSC_DECIDE, 1));
  PetscCall(VecSetFromOptions(x));
  PetscCall(VecDuplicate(x, &r));

  PetscCall(MatCreate(PETSC_COMM_WORLD, &J));
  PetscCall(MatSetSizes(J, PETSC_DECIDE, PETSC_DECIDE, 1, 1));
  PetscCall(MatSetFromOptions(J));
  PetscCall(MatSetUp(J));

  PetscCall(SNESCreate(PETSC_COMM_WORLD, &snes));
  PetscCall(SNESSetFunction(snes, r, FormFunction, NULL));
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, &appctx));
  PetscCall(SNESSetNPCSide(snes, PC_LEFT));
  PetscCall(SNESGetNPC(snes, &npc));
  PetscCall(SNESSetType(npc, SNESNEWTONLS));

  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSetUp(snes));

  PetscCall(VecSet(x, 1.0));
  PetscCall(SNESComputeJacobian(snes, x, J, J));

  PetscCheck(appctx.jac_calls > 0, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Jacobian function was not called with left NPC");
  PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Jacobian called with left NPC\n"));

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: left_npc

TEST*/
