
/*
    Tests PCFIELDSPLIT and hence VecGetRestoreArray_Nest() usage in VecScatter

    Example contributed by: Mike Wick <michael.wick.1980@gmail.com>
*/
#include "petscksp.h"

int main(int argc, char **argv)
{
  Mat         A;
  Mat         subA[9];
  IS          isg[3];
  PetscInt    row, col, mstart, mend;
  PetscScalar val;
  Vec         subb[3];
  Vec         b;
  Vec         r;
  KSP         ksp;
  PC          pc;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, NULL));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 5, 5, PETSC_DETERMINE, PETSC_DETERMINE, 3, NULL, 0, NULL, &subA[0]));
  PetscCall(MatGetOwnershipRange(subA[0], &mstart, &mend));
  for (row = mstart; row < mend; ++row) {
    val = 1.0;
    col = row;
    PetscCall(MatSetValues(subA[0], 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(subA[0], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(subA[0], MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 5, 3, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 1, NULL, &subA[1]));
  PetscCall(MatGetOwnershipRange(subA[1], &mstart, &mend));
  for (row = mstart; row < mend; ++row) {
    col = 1;
    val = 0.0;
    PetscCall(MatSetValues(subA[1], 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(subA[1], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(subA[1], MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 5, 4, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 1, NULL, &subA[2]));
  PetscCall(MatGetOwnershipRange(subA[2], &mstart, &mend));
  for (row = mstart; row < mend; ++row) {
    col = 1;
    val = 0.0;
    PetscCall(MatSetValues(subA[2], 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(subA[2], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(subA[2], MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 3, 5, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 1, NULL, &subA[3]));
  PetscCall(MatGetOwnershipRange(subA[3], &mstart, &mend));
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 0.0;
    PetscCall(MatSetValues(subA[3], 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(subA[3], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(subA[3], MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 3, 3, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 1, NULL, &subA[4]));
  PetscCall(MatGetOwnershipRange(subA[4], &mstart, &mend));
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 4.0;
    PetscCall(MatSetValues(subA[4], 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(subA[4], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(subA[4], MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 3, 4, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 1, NULL, &subA[5]));
  PetscCall(MatGetOwnershipRange(subA[5], &mstart, &mend));
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 0.0;
    PetscCall(MatSetValues(subA[5], 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(subA[5], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(subA[5], MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 4, 5, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 1, NULL, &subA[6]));
  PetscCall(MatGetOwnershipRange(subA[6], &mstart, &mend));
  for (row = mstart; row < mend; ++row) {
    col = 2;
    val = 0.0;
    PetscCall(MatSetValues(subA[6], 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(subA[6], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(subA[6], MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 4, 3, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 1, NULL, &subA[7]));
  PetscCall(MatGetOwnershipRange(subA[7], &mstart, &mend));
  for (row = mstart; row < mend; ++row) {
    col = 1;
    val = 0.0;
    PetscCall(MatSetValues(subA[7], 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(subA[7], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(subA[7], MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, 4, 4, PETSC_DETERMINE, PETSC_DETERMINE, 1, NULL, 1, NULL, &subA[8]));
  PetscCall(MatGetOwnershipRange(subA[8], &mstart, &mend));
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 8.0;
    PetscCall(MatSetValues(subA[8], 1, &row, 1, &col, &val, INSERT_VALUES));
  }
  PetscCall(MatAssemblyBegin(subA[8], MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(subA[8], MAT_FINAL_ASSEMBLY));

  PetscCall(MatCreateNest(PETSC_COMM_WORLD, 3, NULL, 3, NULL, subA, &A));
  PetscCall(MatNestGetISs(A, isg, NULL));
  PetscCall(VecCreate(PETSC_COMM_WORLD, &subb[0]));
  PetscCall(VecSetSizes(subb[0], 5, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(subb[0]));
  PetscCall(VecSet(subb[0], 1.0));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &subb[1]));
  PetscCall(VecSetSizes(subb[1], 3, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(subb[1]));
  PetscCall(VecSet(subb[1], 2.0));

  PetscCall(VecCreate(PETSC_COMM_WORLD, &subb[2]));
  PetscCall(VecSetSizes(subb[2], 4, PETSC_DECIDE));
  PetscCall(VecSetFromOptions(subb[2]));
  PetscCall(VecSet(subb[2], 3.0));

  PetscCall(VecCreateNest(PETSC_COMM_WORLD, 3, NULL, subb, &b));
  PetscCall(VecDuplicate(b, &r));
  PetscCall(VecCopy(b, r));

  PetscCall(MatMult(A, b, r));
  PetscCall(VecSet(b, 0.0));

  PetscCall(KSPCreate(PETSC_COMM_WORLD, &ksp));
  PetscCall(KSPSetOperators(ksp, A, A));
  PetscCall(KSPSetFromOptions(ksp));
  PetscCall(KSPGetPC(ksp, &pc));
  PetscCall(PCFieldSplitSetIS(pc, "a", isg[0]));
  PetscCall(PCFieldSplitSetIS(pc, "b", isg[1]));
  PetscCall(PCFieldSplitSetIS(pc, "c", isg[2]));

  PetscCall(KSPSolve(ksp, r, b));
  PetscCall(KSPView(ksp, PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(MatDestroy(&subA[0]));
  PetscCall(MatDestroy(&subA[1]));
  PetscCall(MatDestroy(&subA[2]));
  PetscCall(MatDestroy(&subA[3]));
  PetscCall(MatDestroy(&subA[4]));
  PetscCall(MatDestroy(&subA[5]));
  PetscCall(MatDestroy(&subA[6]));
  PetscCall(MatDestroy(&subA[7]));
  PetscCall(MatDestroy(&subA[8]));
  PetscCall(MatDestroy(&A));
  PetscCall(VecDestroy(&subb[0]));
  PetscCall(VecDestroy(&subb[1]));
  PetscCall(VecDestroy(&subb[2]));
  PetscCall(VecDestroy(&b));
  PetscCall(VecDestroy(&r));
  PetscCall(KSPDestroy(&ksp));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

    test:
      args: -pc_type fieldsplit -ksp_monitor

TEST*/
