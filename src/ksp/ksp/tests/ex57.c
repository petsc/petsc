
/*
    Tests PCFIELDSPLIT and hence VecGetRestoreArray_Nest() usage in VecScatter

    Example contributed by: Mike Wick <michael.wick.1980@gmail.com>
*/
#include "petscksp.h"

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  Mat            A;
  Mat            subA[9];
  IS             isg[3];
  PetscInt       row,col,mstart,mend;
  PetscScalar    val;
  Vec            subb[3];
  Vec            b;
  Vec            r;
  KSP           ksp;
  PC            pc;

  ierr = PetscInitialize(&argc,&argv,(char*)0,PETSC_NULL);if (ierr) return ierr;

  CHKERRQ(MatCreateAIJ(PETSC_COMM_WORLD,5,5,PETSC_DETERMINE,PETSC_DETERMINE,3,NULL,0,NULL,&subA[0]));
  CHKERRQ(MatGetOwnershipRange(subA[0],&mstart,&mend));
  for (row = mstart; row < mend; ++row) {
    val = 1.0; col = row;
    CHKERRQ(MatSetValues(subA[0],1,&row,1,&col,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(subA[0],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(subA[0],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateAIJ( PETSC_COMM_WORLD,5,3,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[1]));
  CHKERRQ(MatGetOwnershipRange(subA[1],&mstart,&mend));
  for (row = mstart; row < mend; ++row) {
    col = 1;
    val = 0.0;
    CHKERRQ(MatSetValues(subA[1],1,&row,1,&col,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(subA[1],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(subA[1],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateAIJ( PETSC_COMM_WORLD,5,4,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[2]));
  CHKERRQ(MatGetOwnershipRange(subA[2],&mstart,&mend));
  for (row = mstart; row < mend; ++row) {
    col = 1;
    val = 0.0;
    CHKERRQ(MatSetValues(subA[2],1,&row,1,&col,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(subA[2],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(subA[2],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateAIJ( PETSC_COMM_WORLD,3,5,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[3]));
  CHKERRQ(MatGetOwnershipRange(subA[3],&mstart,&mend));
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 0.0;
    CHKERRQ(MatSetValues(subA[3],1,&row,1,&col,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(subA[3],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(subA[3],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateAIJ( PETSC_COMM_WORLD,3,3,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[4]));
  CHKERRQ(MatGetOwnershipRange(subA[4],&mstart,&mend));
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 4.0;
    CHKERRQ(MatSetValues(subA[4],1,&row,1,&col,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(subA[4],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(subA[4],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateAIJ( PETSC_COMM_WORLD,3,4,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[5]));
  CHKERRQ(MatGetOwnershipRange(subA[5],&mstart,&mend));
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 0.0;
    CHKERRQ(MatSetValues(subA[5],1,&row,1,&col,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(subA[5],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(subA[5],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateAIJ( PETSC_COMM_WORLD,4,5,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[6]));
  CHKERRQ(MatGetOwnershipRange(subA[6],&mstart,&mend));
  for (row = mstart; row < mend; ++row) {
    col = 2;
    val = 0.0;
    CHKERRQ(MatSetValues(subA[6],1,&row,1,&col,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(subA[6],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(subA[6],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateAIJ( PETSC_COMM_WORLD,4,3,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[7]));
  CHKERRQ(MatGetOwnershipRange(subA[7],&mstart,&mend));
  for (row = mstart; row < mend; ++row) {
    col = 1;
    val = 0.0;
    CHKERRQ(MatSetValues(subA[7],1,&row,1,&col,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(subA[7],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(subA[7],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateAIJ( PETSC_COMM_WORLD,4,4,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[8]));
  CHKERRQ(MatGetOwnershipRange(subA[8],&mstart,&mend));
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 8.0;
    CHKERRQ(MatSetValues(subA[8],1,&row,1,&col,&val,INSERT_VALUES));
  }
  CHKERRQ(MatAssemblyBegin(subA[8],MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(subA[8],MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatCreateNest(PETSC_COMM_WORLD,3,NULL,3,NULL,subA,&A));
  CHKERRQ(MatNestGetISs(A,isg,NULL));
  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&subb[0]));
  CHKERRQ(VecSetSizes(subb[0],5,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(subb[0]));
  CHKERRQ(VecSet(subb[0],1.0));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&subb[1]));
  CHKERRQ(VecSetSizes(subb[1],3,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(subb[1]));
  CHKERRQ(VecSet(subb[1],2.0));

  CHKERRQ(VecCreate(PETSC_COMM_WORLD,&subb[2]));
  CHKERRQ(VecSetSizes(subb[2],4,PETSC_DECIDE));
  CHKERRQ(VecSetFromOptions(subb[2]));
  CHKERRQ(VecSet(subb[2],3.0));

  CHKERRQ(VecCreateNest(PETSC_COMM_WORLD,3,NULL,subb,&b));
  CHKERRQ(VecDuplicate(b,&r));
  CHKERRQ(VecCopy(b,r));

  CHKERRQ(MatMult(A,b,r));
  CHKERRQ(VecSet(b,0.0));

  CHKERRQ(KSPCreate(PETSC_COMM_WORLD,&ksp));
  CHKERRQ(KSPSetOperators(ksp,A,A));
  CHKERRQ(KSPSetFromOptions(ksp));
  CHKERRQ(KSPGetPC(ksp,&pc));
  CHKERRQ(PCFieldSplitSetIS(pc,"a",isg[0]));
  CHKERRQ(PCFieldSplitSetIS(pc,"b",isg[1]));
  CHKERRQ(PCFieldSplitSetIS(pc,"c",isg[2]));

  CHKERRQ(KSPSolve(ksp,r,b));
  CHKERRQ(KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&subA[0]));
  CHKERRQ(MatDestroy(&subA[1]));
  CHKERRQ(MatDestroy(&subA[2]));
  CHKERRQ(MatDestroy(&subA[3]));
  CHKERRQ(MatDestroy(&subA[4]));
  CHKERRQ(MatDestroy(&subA[5]));
  CHKERRQ(MatDestroy(&subA[6]));
  CHKERRQ(MatDestroy(&subA[7]));
  CHKERRQ(MatDestroy(&subA[8]));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecDestroy(&subb[0]));
  CHKERRQ(VecDestroy(&subb[1]));
  CHKERRQ(VecDestroy(&subb[2]));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(VecDestroy(&r));
  CHKERRQ(KSPDestroy(&ksp));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -pc_type fieldsplit -ksp_monitor

TEST*/
