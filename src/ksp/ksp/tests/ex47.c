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
  CHKERRQ(MatCreate(comm, &A));
  CHKERRQ(MatSetSizes(A, 4, 4, PETSC_DECIDE, PETSC_DECIDE));
  CHKERRQ(MatSetUp(A));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatCreateVecs(A, &x, &b));
  CHKERRQ(VecSet(x, 2.0));
  CHKERRQ(VecSet(b, 12.0));
  CHKERRQ(MatDiagonalSet(A, x, INSERT_VALUES));
  CHKERRQ(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  CHKERRQ(ISCreateStride(comm, 2, 0, 1, &zero));
  CHKERRQ(ISCreateStride(comm, 2, 2, 1, &one));
  CHKERRQ(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullsp));
  CHKERRQ(PetscObjectCompose((PetscObject)zero, "nullspace",(PetscObject)nullsp));
  CHKERRQ(KSPCreate(comm, &ksp));
  CHKERRQ(KSPSetOperators(ksp, A, A));
  CHKERRQ(KSPSetUp(ksp));
  CHKERRQ(KSPGetPC(ksp, &pc));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(PCFieldSplitSetIS(pc, "0", zero));
  CHKERRQ(PCFieldSplitSetIS(pc, "1", one));
  CHKERRQ(KSPSolve(ksp, b, x));
  CHKERRQ(KSPDestroy(&ksp));
  CHKERRQ(MatNullSpaceDestroy(&nullsp));
  CHKERRQ(ISDestroy(&zero));
  CHKERRQ(ISDestroy(&one));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&b));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      args: -pc_type fieldsplit

TEST*/
