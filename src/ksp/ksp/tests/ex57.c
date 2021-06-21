
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

  ierr = MatCreateAIJ(PETSC_COMM_WORLD,5,5,PETSC_DETERMINE,PETSC_DETERMINE,3,NULL,0,NULL,&subA[0]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(subA[0],&mstart,&mend);CHKERRQ(ierr);
  for (row = mstart; row < mend; ++row) {
    val = 1.0; col = row;
    ierr = MatSetValues(subA[0],1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(subA[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA[0],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateAIJ( PETSC_COMM_WORLD,5,3,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[1]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(subA[1],&mstart,&mend);CHKERRQ(ierr);
  for (row = mstart; row < mend; ++row) {
    col = 1;
    val = 0.0;
    ierr = MatSetValues(subA[1],1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(subA[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA[1],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateAIJ( PETSC_COMM_WORLD,5,4,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[2]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(subA[2],&mstart,&mend);CHKERRQ(ierr);
  for (row = mstart; row < mend; ++row) {
    col = 1;
    val = 0.0;
    ierr = MatSetValues(subA[2],1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(subA[2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA[2],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateAIJ( PETSC_COMM_WORLD,3,5,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[3]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(subA[3],&mstart,&mend);CHKERRQ(ierr);
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 0.0;
    ierr = MatSetValues(subA[3],1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(subA[3],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA[3],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateAIJ( PETSC_COMM_WORLD,3,3,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[4]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(subA[4],&mstart,&mend);CHKERRQ(ierr);
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 4.0;
    ierr = MatSetValues(subA[4],1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(subA[4],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA[4],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateAIJ( PETSC_COMM_WORLD,3,4,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[5]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(subA[5],&mstart,&mend);CHKERRQ(ierr);
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 0.0;
    ierr = MatSetValues(subA[5],1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(subA[5],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA[5],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateAIJ( PETSC_COMM_WORLD,4,5,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[6]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(subA[6],&mstart,&mend);CHKERRQ(ierr);
  for (row = mstart; row < mend; ++row) {
    col = 2;
    val = 0.0;
    ierr = MatSetValues(subA[6],1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(subA[6],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA[6],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateAIJ( PETSC_COMM_WORLD,4,3,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[7]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(subA[7],&mstart,&mend);CHKERRQ(ierr);
  for (row = mstart; row < mend; ++row) {
    col = 1;
    val = 0.0;
    ierr = MatSetValues(subA[7],1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(subA[7],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA[7],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateAIJ( PETSC_COMM_WORLD,4,4,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,1,NULL,&subA[8]);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(subA[8],&mstart,&mend);CHKERRQ(ierr);
  for (row = mstart; row < mend; ++row) {
    col = row;
    val = 8.0;
    ierr = MatSetValues(subA[8],1,&row,1,&col,&val,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(subA[8],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(subA[8],MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatCreateNest(PETSC_COMM_WORLD,3,NULL,3,NULL,subA,&A);CHKERRQ(ierr);
  ierr = MatNestGetISs(A,isg,NULL);CHKERRQ(ierr);
  ierr = VecCreate(PETSC_COMM_WORLD,&subb[0]);CHKERRQ(ierr);
  ierr = VecSetSizes(subb[0],5,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(subb[0]);CHKERRQ(ierr);
  ierr = VecSet(subb[0],1.0);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&subb[1]);CHKERRQ(ierr);
  ierr = VecSetSizes(subb[1],3,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(subb[1]);CHKERRQ(ierr);
  ierr = VecSet(subb[1],2.0);CHKERRQ(ierr);

  ierr = VecCreate(PETSC_COMM_WORLD,&subb[2]);CHKERRQ(ierr);
  ierr = VecSetSizes(subb[2],4,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = VecSetFromOptions(subb[2]);CHKERRQ(ierr);
  ierr = VecSet(subb[2],3.0);CHKERRQ(ierr);

  ierr = VecCreateNest(PETSC_COMM_WORLD,3,NULL,subb,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(b,&r);CHKERRQ(ierr);
  ierr = VecCopy(b,r);CHKERRQ(ierr);

  ierr = MatMult(A,b,r);CHKERRQ(ierr);
  ierr = VecSet(b,0.0);CHKERRQ(ierr);

  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp,A,A);CHKERRQ(ierr);
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp,&pc);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc,"a",isg[0]);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc,"b",isg[1]);CHKERRQ(ierr);
  ierr = PCFieldSplitSetIS(pc,"c",isg[2]);CHKERRQ(ierr);

  ierr = KSPSolve(ksp,r,b);CHKERRQ(ierr);
  ierr = KSPView(ksp,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&subA[0]);CHKERRQ(ierr);
  ierr = MatDestroy(&subA[1]);CHKERRQ(ierr);
  ierr = MatDestroy(&subA[2]);CHKERRQ(ierr);
  ierr = MatDestroy(&subA[3]);CHKERRQ(ierr);
  ierr = MatDestroy(&subA[4]);CHKERRQ(ierr);
  ierr = MatDestroy(&subA[5]);CHKERRQ(ierr);
  ierr = MatDestroy(&subA[6]);CHKERRQ(ierr);
  ierr = MatDestroy(&subA[7]);CHKERRQ(ierr);
  ierr = MatDestroy(&subA[8]);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecDestroy(&subb[0]);CHKERRQ(ierr);
  ierr = VecDestroy(&subb[1]);CHKERRQ(ierr);
  ierr = VecDestroy(&subb[2]);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = VecDestroy(&r);CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

    test:
      args: -pc_type fieldsplit -ksp_monitor

TEST*/
