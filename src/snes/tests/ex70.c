static char help[] = "Tests that SNESSetUp() propagates application context to the NPC only when the NPC has no existing context.\n\n";

#include <petscsnes.h>

typedef struct {
  PetscInt tag;
} AppCtx;

PetscErrorCode FormFunction(SNES snes, Vec x, Vec f, PetscCtx ctx)
{
  const PetscScalar *xx;
  PetscScalar       *ff;

  PetscFunctionBeginUser;
  PetscCall(VecGetArrayRead(x, &xx));
  PetscCall(VecGetArray(f, &ff));
  ff[0] = xx[0];
  PetscCall(VecRestoreArrayRead(x, &xx));
  PetscCall(VecRestoreArray(f, &ff));
  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode FormJacobian(SNES snes, Vec x, Mat jac, Mat B, PetscCtx ctx)
{
  PetscScalar one = 1.0;
  PetscInt    idx = 0;

  PetscFunctionBeginUser;
  PetscCall(MatSetValues(B, 1, &idx, 1, &idx, &one, INSERT_VALUES));
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
  SNES      snes, npc;
  Vec       x, r;
  Mat       J;
  AppCtx    parent_ctx, npc_ctx;
  void     *retrieved_ctx;
  PetscBool test_preserve = PETSC_FALSE;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, help));
  PetscCall(PetscOptionsGetBool(NULL, NULL, "-test_preserve", &test_preserve, NULL));

  parent_ctx.tag = 1;
  npc_ctx.tag    = 2;

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
  PetscCall(SNESSetJacobian(snes, J, J, FormJacobian, NULL));
  PetscCall(SNESSetApplicationContext(snes, &parent_ctx));

  /* Use SNESSetNPC with a freshly created SNES so no automatic context copy occurs */
  PetscCall(SNESCreate(PETSC_COMM_WORLD, &npc));
  if (test_preserve) PetscCall(SNESSetApplicationContext(npc, &npc_ctx));
  PetscCall(SNESSetNPC(snes, npc));
  PetscCall(SNESDestroy(&npc));

  PetscCall(SNESSetFromOptions(snes));
  PetscCall(SNESSetUp(snes));

  PetscCall(SNESGetNPC(snes, &npc));
  PetscCall(SNESGetApplicationContext(npc, &retrieved_ctx));
  if (test_preserve) {
    PetscCheck(retrieved_ctx == (void *)&npc_ctx, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "NPC context was overwritten by parent context");
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NPC kept its own context (tag=%" PetscInt_FMT ")\n", ((AppCtx *)retrieved_ctx)->tag));
  } else {
    PetscCheck(retrieved_ctx == (void *)&parent_ctx, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "NPC did not inherit parent context");
    PetscCall(PetscPrintf(PETSC_COMM_WORLD, "NPC inherited parent context (tag=%" PetscInt_FMT ")\n", ((AppCtx *)retrieved_ctx)->tag));
  }

  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&r));
  PetscCall(MatDestroy(&J));
  PetscCall(SNESDestroy(&snes));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: inherit
      nsize: 1
      output_file: output/ex70_inherit.out

   test:
      suffix: preserve
      nsize: 1
      args: -test_preserve
      output_file: output/ex70_preserve.out

TEST*/
