/*
    Tests attaching null space to IS for fieldsplit preconditioner
*/
#include <petscksp.h>

int main(int argc, char **argv)
{
  Mat            A;
  KSP            ksp;
  PC             pc;
  IS             zero, one;
  MatNullSpace   nullsp;
  Vec            x, b;
  MPI_Comm       comm;

  PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
  comm = PETSC_COMM_WORLD;
  PetscCall(MatCreate(comm, &A));
  PetscCall(MatSetSizes(A, 4, 4, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetUp(A));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatCreateVecs(A, &x, &b));
  PetscCall(VecSet(x, 2.0));
  PetscCall(VecSet(b, 12.0));
  PetscCall(MatDiagonalSet(A, x, INSERT_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(ISCreateStride(comm, 2, 0, 1, &zero));
  PetscCall(ISCreateStride(comm, 2, 2, 1, &one));
  PetscCall(MatNullSpaceCreate(comm, PETSC_TRUE, 0, NULL, &nullsp));
  PetscCall(PetscObjectCompose((PetscObject)zero, "nullspace",(PetscObject)nullsp));
  PetscCall(KSPCreate(comm, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetUp(ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(PCFieldSplitSetIS(pc, "0", zero));
  PetscCall(PCFieldSplitSetIS(pc, "1", one));
  PetscCall(KSPSolve(ksp, b, x));
  PetscCall(KSPDestroy(&ksp));
  PetscCall(MatNullSpaceDestroy(&nullsp));
  PetscCall(ISDestroy(&zero));
  PetscCall(ISDestroy(&one));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&b));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -pc_type fieldsplit

TEST*/
