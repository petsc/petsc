static char help[] = "Test MatTransposeColoring for SeqAIJ matrices.\n\n";

#include <petscmat.h>
#include <../src/mat/impls/dense/seq/dense.h> /*I "petscmat.h" I*/

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv) {
  Mat                   A,R,C,C_dense,C_sparse,Rt_dense,P,PtAP;
  PetscInt              row,col,m,n;
  PetscErrorCode        ierr;
  MatScalar             one=1.0,val;
  MatTransposeColoring  matcoloring = 0;
  ISColoring            iscoloring;
  PetscBool             equal;

  PetscInitialize(&argc,&argv,(char *)0,help);
  /* Create A */
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,4,4,4,4);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  row=0; col=0; val=1.0; ierr = MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
  row=1; col=3; val=2.0; ierr = MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
  row=2; col=2; val=3.0; ierr = MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
  row=3; col=0; val=4.0; ierr = MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"A_");CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
 
  /* Create R */
  ierr = MatCreate(PETSC_COMM_SELF,&R);CHKERRQ(ierr);
  ierr = MatSetSizes(R,2,4,2,4);CHKERRQ(ierr);
  ierr = MatSetType(R,MATSEQAIJ);CHKERRQ(ierr);
  row=0; col=0; ierr = MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES);CHKERRQ(ierr);
  row=0; col=1; ierr = MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES);CHKERRQ(ierr);

  row=1; col=1; ierr = MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES);CHKERRQ(ierr);
  row=1; col=2; ierr = MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES);CHKERRQ(ierr);
  row=1; col=3; ierr = MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(R,"R_");CHKERRQ(ierr);
  ierr = MatView(R,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

  /* C = A*R^T */
  ierr = MatMatTransposeMult(A,R,MAT_INITIAL_MATRIX,2.0,&C);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C,"ARt_");CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);
 
  /* Create MatTransposeColoring from symbolic C=A*R^T */
  ierr = MatGetColoring(C,MATCOLORINGSL,&iscoloring);CHKERRQ(ierr); 
  ierr = MatTransposeColoringCreate(C,iscoloring,&matcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

  /* Create Rt_dense */
  ierr = MatCreate(PETSC_COMM_WORLD,&Rt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(Rt_dense,A->cmap->n,matcoloring->ncolors,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Rt_dense,MATDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Rt_dense,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Rt_dense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Rt_dense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Rt_dense,&m,&n);CHKERRQ(ierr);
  printf("Rt_dense: %d,%d\n",m,n);

  /* Get Rt_dense by Apply MatTransposeColoring to R */
  ierr = MatTransColoringApplySpToDen(matcoloring,R,Rt_dense);CHKERRQ(ierr);

  /* C_dense = A*Rt_dense */
  ierr = MatMatMult(A,Rt_dense,MAT_INITIAL_MATRIX,2.0,&C_dense);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C_dense,"ARt_dense_");CHKERRQ(ierr);
  //ierr = MatView(C_dense,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  //ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

  /* Recover C from C_dense */
  ierr = MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&C_sparse);CHKERRQ(ierr);
  ierr = MatTransColoringApplyDenToSp(matcoloring,C_dense,C_sparse);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C_sparse,"ARt_color_");CHKERRQ(ierr);
  ierr = MatView(C_sparse,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

  ierr = MatDestroy(&C_dense);CHKERRQ(ierr);
  ierr = MatDestroy(&C_sparse);CHKERRQ(ierr);
  ierr = MatDestroy(&Rt_dense);CHKERRQ(ierr);
  ierr = MatTransposeColoringDestroy(&matcoloring);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr); 

  /* Test PtAP = P^T*A*P, P = R^T */
  ierr = MatTranspose(R,MAT_INITIAL_MATRIX,&P);CHKERRQ(ierr);
  ierr = MatPtAP(A,P,MAT_INITIAL_MATRIX,2.0,&PtAP);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(PtAP,"PtAP_");CHKERRQ(ierr);
  ierr = MatView(PtAP,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = MatDestroy(&P);CHKERRQ(ierr);

  /* Test C = RARt */
  ierr = MatRARt(A,R,MAT_INITIAL_MATRIX,2.0,&C);CHKERRQ(ierr);
  ierr = MatEqual(C,PtAP,&equal);CHKERRQ(ierr);
  if (!equal) {
    ierr = PetscPrintf(PETSC_COMM_SELF,"Error: PtAP != RARt");CHKERRQ(ierr);
  }

  /* Free spaces */
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&PtAP);CHKERRQ(ierr);

  PetscFinalize();
  return(0);
}
