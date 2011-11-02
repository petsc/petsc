static char help[] = "Test MatMultTransposeColoring for AIJ matrices.\n\n";

#include <petscmat.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat            A,B,C,C_dense,C_sparse,Bt_dense;
  PetscInt       I,J,m,n;
  PetscErrorCode ierr;
  MatScalar      one=1.0,val;
  MatTransposeColoring  matcoloring = 0;
  ISColoring            iscoloring;

  PetscInitialize(&argc,&argv,(char *)0,help);
  /* Create A */
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,4,4,4,4);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  I=0; J=0; val=1.0; ierr = MatSetValues(A,1,&I,1,&J,&val,ADD_VALUES);CHKERRQ(ierr);
  I=1; J=3; val=2.0; ierr = MatSetValues(A,1,&I,1,&J,&val,ADD_VALUES);CHKERRQ(ierr);
  I=2; J=2; val=3.0; ierr = MatSetValues(A,1,&I,1,&J,&val,ADD_VALUES);CHKERRQ(ierr);
  I=3; J=0; val=4.0; ierr = MatSetValues(A,1,&I,1,&J,&val,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"A_");CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
 
  /* Create B */
  ierr = MatCreate(PETSC_COMM_SELF,&B);CHKERRQ(ierr);
  ierr = MatSetSizes(B,2,4,2,4);CHKERRQ(ierr);
  ierr = MatSetType(B,MATSEQAIJ);CHKERRQ(ierr);
  I=0; J=0; ierr = MatSetValues(B,1,&I,1,&J,&one,ADD_VALUES);CHKERRQ(ierr);
  I=0; J=1; ierr = MatSetValues(B,1,&I,1,&J,&one,ADD_VALUES);CHKERRQ(ierr);

  I=1; J=1; ierr = MatSetValues(B,1,&I,1,&J,&one,ADD_VALUES);CHKERRQ(ierr);
  I=1; J=2; ierr = MatSetValues(B,1,&I,1,&J,&one,ADD_VALUES);CHKERRQ(ierr);
  I=1; J=3; ierr = MatSetValues(B,1,&I,1,&J,&one,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(B,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(B,"B_");CHKERRQ(ierr);
  ierr = MatView(B,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

  /* C = A*B^T */
  ierr = MatMatTransposeMult(A,B,MAT_INITIAL_MATRIX,2.0,&C);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C,"C_");CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
 
  /* Create MatTransposeColoring from symbolic C=A*B^T */
  ierr = MatGetColoring(C,MATCOLORINGSL,&iscoloring);CHKERRQ(ierr); 
  ierr = MatTransposeColoringCreate(C,iscoloring,&matcoloring);CHKERRQ(ierr);
  //ierr = MatFDColoringSetFromOptions(matcoloring);CHKERRQ(ierr);
  //ierr = MatFDColoringView((MatFDColoring)matcoloring,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  //ierr = MatFDColoringView(matcoloring,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

  /* Create Bt_dense */
  ierr = MatCreate(PETSC_COMM_WORLD,&Bt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(Bt_dense,A->cmap->n,matcoloring->ncolors,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Bt_dense,MATDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Bt_dense,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Bt_dense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Bt_dense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Bt_dense,&m,&n);CHKERRQ(ierr);
  printf("Bt_dense: %d,%d\n",m,n);

  /* Get Bt_dense by Apply MatTransposeColoring to B */
  ierr = MatTransColoringApplySpToDen(matcoloring,B,Bt_dense);CHKERRQ(ierr);

  /* C_dense = A*Bt_dense */
  ierr = MatMatMult(A,Bt_dense,MAT_INITIAL_MATRIX,2.0,&C_dense);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C_dense,"C_dense_");CHKERRQ(ierr);
  //ierr = MatView(C_dense,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

  /* Recover C from C_dense */
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&C_sparse);CHKERRQ(ierr);
  ierr = MatTransColoringApplyDenToSp(matcoloring,C_dense,C_sparse);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C_sparse,"C_sparse_");CHKERRQ(ierr);
  ierr = MatView(C_sparse,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
 
  /* Free spaces */
  ierr = MatDestroy(&C_dense);CHKERRQ(ierr);
  ierr = MatDestroy(&C_sparse);CHKERRQ(ierr);
  ierr = MatDestroy(&Bt_dense);CHKERRQ(ierr);
  ierr = MatTransposeColoringDestroy(&matcoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr); 
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&B);CHKERRQ(ierr);
  PetscFinalize();
  return(0);
}
