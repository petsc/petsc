/*
    Tests attaching null space to IS for fieldsplit preconditioner
*/
#include <petscksp.h>

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  Mat            A;
  KSP            ksp;
  PC             pc;
  IS             zero, one;
  MatNullSpace   nullsp;
  Vec            x, b;
  MPI_Comm       comm;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL);if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = MatCreate(comm, &A);CHKERRQ(ierr);
  ierr = MatSetSizes(A, 4, 4, PETSC_DECIDE, PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatCreateVecs(A, &x, &b);CHKERRQ(ierr);
  ierr = VecSet(x, 2.0);CHKERRQ(ierr);
  ierr = VecSet(b, 12.0);CHKERRQ(ierr);
  ierr = MatDiagonalSet(A, x, INSERT_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = ISCreateStride(comm, 2, 0, 1, &zero);CHKERRQ(ierr);
  ierr = ISCreateStride(comm, 2, 2, 1, &one);CHKERRQ(ierr);
  ierr = MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullsp);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)zero, "nullspace",(PetscObject)nullsp);CHKERRQ(ierr);
  ierr = KSPCreate(comm, &ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, A, A);CHKERRQ(ierr);
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc, "0", zero);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc, "1", one);CHKERRQ(ierr);
  ierr = KSPSolve(ksp, b, x);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = MatNullSpaceDestroy(&nullsp);CHKERRQ(ierr);
  ierr = ISDestroy(&zero);CHKERRQ(ierr);
  ierr = ISDestroy(&one);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      args: -pc_type fieldsplit

TEST*/
