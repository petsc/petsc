static char help[] = "Test MatTransposeColoring for SeqAIJ matrices. Used for '-matmattransmult_color' on  MatMatTransposeMult \n\n";

#include <petscmat.h>
#include <petsc/private/matimpl.h> /* Need struct _p_MatTransposeColoring for this test. */

int main(int argc,char **argv)
{
  Mat                  A,R,C,C_dense,C_sparse,Rt_dense,P,PtAP;
  PetscInt             row,col,m,n;
  PetscErrorCode       ierr;
  MatScalar            one         =1.0,val;
  MatColoring          mc;
  MatTransposeColoring matcoloring = 0;
  ISColoring           iscoloring;
  PetscBool            equal;
  PetscMPIInt          size;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* Create seqaij A */
  ierr = MatCreate(PETSC_COMM_SELF,&A);CHKERRQ(ierr);
  ierr = MatSetSizes(A,4,4,4,4);CHKERRQ(ierr);
  ierr = MatSetType(A,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(A);CHKERRQ(ierr);
  ierr = MatSetUp(A);CHKERRQ(ierr);
  row  = 0; col=0; val=1.0; ierr = MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
  row  = 1; col=3; val=2.0; ierr = MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
  row  = 2; col=2; val=3.0; ierr = MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
  row  = 3; col=0; val=4.0; ierr = MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(A,"A_");CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr);

  /* Create seqaij R */
  ierr = MatCreate(PETSC_COMM_SELF,&R);CHKERRQ(ierr);
  ierr = MatSetSizes(R,2,4,2,4);CHKERRQ(ierr);
  ierr = MatSetType(R,MATSEQAIJ);CHKERRQ(ierr);
  ierr = MatSetFromOptions(R);CHKERRQ(ierr);
  ierr = MatSetUp(R);CHKERRQ(ierr);
  row  = 0; col=0; ierr = MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES);CHKERRQ(ierr);
  row  = 0; col=1; ierr = MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES);CHKERRQ(ierr);

  row  = 1; col=1; ierr = MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES);CHKERRQ(ierr);
  row  = 1; col=2; ierr = MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES);CHKERRQ(ierr);
  row  = 1; col=3; ierr = MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES);CHKERRQ(ierr);
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
  ierr = MatColoringCreate(C,&mc);CHKERRQ(ierr);
  ierr = MatColoringSetDistance(mc,2);CHKERRQ(ierr);
  /* ierr = MatColoringSetType(mc,MATCOLORINGSL);CHKERRQ(ierr); */
  ierr = MatColoringSetFromOptions(mc);CHKERRQ(ierr);
  ierr = MatColoringApply(mc,&iscoloring);CHKERRQ(ierr);
  ierr = MatColoringDestroy(&mc);CHKERRQ(ierr);
  ierr = MatTransposeColoringCreate(C,iscoloring,&matcoloring);CHKERRQ(ierr);
  ierr = ISColoringDestroy(&iscoloring);CHKERRQ(ierr);

  /* Create Rt_dense */
  ierr = MatCreate(PETSC_COMM_WORLD,&Rt_dense);CHKERRQ(ierr);
  ierr = MatSetSizes(Rt_dense,4,matcoloring->ncolors,PETSC_DECIDE,PETSC_DECIDE);CHKERRQ(ierr);
  ierr = MatSetType(Rt_dense,MATDENSE);CHKERRQ(ierr);
  ierr = MatSeqDenseSetPreallocation(Rt_dense,NULL);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(Rt_dense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(Rt_dense,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatGetLocalSize(Rt_dense,&m,&n);CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_SELF,"Rt_dense: %D,%D\n",m,n);CHKERRQ(ierr);

  /* Get Rt_dense by Apply MatTransposeColoring to R */
  ierr = MatTransColoringApplySpToDen(matcoloring,R,Rt_dense);CHKERRQ(ierr);

  /* C_dense = A*Rt_dense */
  ierr = MatMatMult(A,Rt_dense,MAT_INITIAL_MATRIX,2.0,&C_dense);CHKERRQ(ierr);
  ierr = MatSetOptionsPrefix(C_dense,"ARt_dense_");CHKERRQ(ierr);
  /*ierr = MatView(C_dense,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr); */
  /*ierr = PetscPrintf(PETSC_COMM_SELF,"\n");CHKERRQ(ierr); */

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
  ierr = MatRARt(A,R,MAT_REUSE_MATRIX,2.0,&C);CHKERRQ(ierr);
  ierr = MatEqual(C,PtAP,&equal);CHKERRQ(ierr);
  if (!equal) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: PtAP != RARt");

  /* Free spaces */
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&R);CHKERRQ(ierr);
  ierr = MatDestroy(&PtAP);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      output_file: output/ex161.out
   test:
      suffix: 2
      args: -matmattransmult_via color
      output_file: output/ex161.out

   test:
      suffix: 3
      args: -matmattransmult_via color -matden2sp_brows 3
      output_file: output/ex161.out

   test:
      suffix: 4
      args: -matmattransmult_via color -matrart_via r*art
      output_file: output/ex161.out

   test:
      suffix: 5
      args: -matmattransmult_via color -matrart_via coloring_rart
      output_file: output/ex161.out

TEST*/
