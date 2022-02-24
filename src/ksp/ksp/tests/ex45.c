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
    Mat            A;
    KSP            ksp;
    DM             shell;
    Vec            *left, *right;
    MPI_Comm       c;
    PetscErrorCode ierr;

    ierr = PetscInitialize(&argc, &argv, NULL, NULL);if (ierr) return ierr;
    c = PETSC_COMM_WORLD;

    CHKERRQ(MatCreate(c, &A));
    CHKERRQ(MatSetSizes(A, 1, 1, PETSC_DECIDE, PETSC_DECIDE));
    CHKERRQ(MatSetFromOptions(A));
    CHKERRQ(MatSetUp(A));
    CHKERRQ(KSPCreate(c, &ksp));
    CHKERRQ(KSPSetOperators(ksp, A, A));
    CHKERRQ(KSPSetFromOptions(ksp));
    CHKERRQ(DMShellCreate(c, &shell));
    CHKERRQ(DMSetFromOptions(shell));
    CHKERRQ(DMSetUp(shell));
    CHKERRQ(KSPSetDM(ksp, shell));

    CHKERRQ(KSPCreateVecs(ksp, 1, &right, 1, &left));
    CHKERRQ(VecView(right[0], PETSC_VIEWER_STDOUT_(c)));
    CHKERRQ(VecDestroyVecs(1,&right));
    CHKERRQ(VecDestroyVecs(1,&left));

    CHKERRQ(DMDestroy(&shell));
    CHKERRQ(KSPDestroy(&ksp));
    CHKERRQ(MatDestroy(&A));
    PetscFinalize();
    return 0;
}

/*TEST

   test:

TEST*/
