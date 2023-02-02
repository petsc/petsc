/*
   Creates a DMShell and uses it with a KSP
   This tests that the KSP object can still create vectors using the Mat object

   Contributed by Lawrence Mitchell as part of pull request 221

*/
#include <petscdm.h>
#include <petscdmshell.h>
#include <petscksp.h>

int main(int argc, char **argv)
{
  Mat      A;
  KSP      ksp;
  DM       shell;
  Vec     *left, *right;
  MPI_Comm c;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  c = PETSC_COMM_WORLD;

  PetscCall(MatCreate(c, &A));
  PetscCall(MatSetSizes(A, 1, 1, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  PetscCall(KSPCreate(c, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(DMShellCreate(c, &shell));
  PetscCall(DMSetFromOptions(shell));
  PetscCall(DMSetUp(shell));
  PetscCall(KSPSetDM(ksp, shell));

  PetscCall(KSPCreateVecs(ksp, 1, &right, 1, &left));
  PetscCall(VecView(right[0], PETSC_VIEWER_STDOUT_(c)));
  PetscCall(VecDestroyVecs(1, &right));
  PetscCall(VecDestroyVecs(1, &left));

  PetscCall(DMDestroy(&shell));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatDestroy(&A));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
