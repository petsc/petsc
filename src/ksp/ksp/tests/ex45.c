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

    ierr = MatCreate(c, &A);CHKERRQ(ierr);
    ierr = MatSetSizes(A, 1, 1, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
    ierr = MatSetFromOptions(A);CHKERRQ(ierr);
    ierr = MatSetUp(A);CHKERRQ(ierr);
    ierr = KSPCreate(c, &ksp);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
    ierr = DMShellCreate(c, &shell);CHKERRQ(ierr);
    ierr = DMSetFromOptions(shell);CHKERRQ(ierr);
    ierr = DMSetUp(shell);CHKERRQ(ierr);
    ierr = KSPSetDM(ksp, shell);CHKERRQ(ierr);

    ierr = KSPCreateVecs(ksp, 1, &right, 1, &left);CHKERRQ(ierr);
    ierr = VecView(right[0], PETSC_VIEWER_STDOUT_(c));CHKERRQ(ierr);
    ierr = VecDestroyVecs(1,&right);CHKERRQ(ierr);
    ierr = VecDestroyVecs(1,&left);CHKERRQ(ierr);

    ierr = DMDestroy(&shell);CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
    PetscFinalize();
    return 0;
}

/*TEST

   test:

TEST*/
