static char help[] = "Tests MATLMVM classes.\n\n";

#include <petscksp.h>

int main(int argc, char **args)
{
  PC          P;
  Mat         A;
  Vec         x, x2, f, u, b;
  PetscInt    n = 4, nup = 10;
  PetscRandom rand;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &args, NULL, help));

  /* make sure LMVM classes are registered */
  PetscCall(KSPInitializePackage());

  /* create matrix */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &A));
  PetscCall(MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n));
  PetscCall(MatSetType(A, MATLMVMBFGS));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));

  /* create preconditioner wrapping MatSolve */
  PetscCall(PCCreate(PETSC_COMM_WORLD, &P));
  PetscCall(PCSetType(P, PCMAT));
  PetscCall(PCSetOperators(P, A, A));
  PetscCall(PCSetUp(P));
  PetscCall(PCView(P, NULL));

  PetscCall(PetscRandomCreate(PETSC_COMM_WORLD, &rand));
  PetscCall(PetscRandomSetType(rand, PETSCRANDER48));

  /* create vectors */
  PetscCall(MatCreateVecs(A, &x, &f));
  PetscCall(VecDuplicate(x, &x2));
  PetscCall(VecDuplicate(x, &u));
  PetscCall(VecDuplicate(f, &b));
  PetscCall(VecSetRandom(u, rand));

  /* Test various routines */
  for (PetscInt i = 0; i < nup; i++) {
    PetscReal err;

    PetscCall(VecSetRandom(x, rand));
    PetscCall(VecSetRandom(f, rand));
    PetscCall(MatLMVMUpdate(A, x, f));
    PetscCall(MatView(A, NULL));
    PetscCall(MatMult(A, u, b));
    PetscCall(MatSolve(A, b, x));
    PetscCall(PCApply(P, b, x2));
    PetscCall(VecAXPY(x2, -1.0, x));
    PetscCall(VecNorm(x2, NORM_2, &err));
    if (err > PETSC_SMALL) PetscCall(PetscPrintf(PETSC_COMM_WORLD, "Error %" PetscInt_FMT ": %g\n", i, (double)err));
  }

  /* cleanup */
  PetscCall(PetscRandomDestroy(&rand));
  PetscCall(VecDestroy(&u));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&x2));
  PetscCall(VecDestroy(&f));
  PetscCall(MatDestroy(&A));
  PetscCall(PCDestroy(&P));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      requires: !complex
      args: -mat_type {{lmvmdfp lmvmbfgs lmvmsr1 lmvmbroyden lmvmbadbroyden lmvmsymbroyden lmvmsymbadbroyden lmvmdiagbroyden}separate output}

TEST*/
