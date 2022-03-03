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
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_SUP,"This is a uniprocessor example only!");

  /* Create seqaij A */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&A));
  CHKERRQ(MatSetSizes(A,4,4,4,4));
  CHKERRQ(MatSetType(A,MATSEQAIJ));
  CHKERRQ(MatSetFromOptions(A));
  CHKERRQ(MatSetUp(A));
  row  = 0; col=0; val=1.0; CHKERRQ(MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES));
  row  = 1; col=3; val=2.0; CHKERRQ(MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES));
  row  = 2; col=2; val=3.0; CHKERRQ(MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES));
  row  = 3; col=0; val=4.0; CHKERRQ(MatSetValues(A,1,&row,1,&col,&val,ADD_VALUES));
  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOptionsPrefix(A,"A_"));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n"));

  /* Create seqaij R */
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&R));
  CHKERRQ(MatSetSizes(R,2,4,2,4));
  CHKERRQ(MatSetType(R,MATSEQAIJ));
  CHKERRQ(MatSetFromOptions(R));
  CHKERRQ(MatSetUp(R));
  row  = 0; col=0; CHKERRQ(MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES));
  row  = 0; col=1; CHKERRQ(MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES));

  row  = 1; col=1; CHKERRQ(MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES));
  row  = 1; col=2; CHKERRQ(MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES));
  row  = 1; col=3; CHKERRQ(MatSetValues(R,1,&row,1,&col,&one,ADD_VALUES));
  CHKERRQ(MatAssemblyBegin(R,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(R,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatSetOptionsPrefix(R,"R_"));
  CHKERRQ(MatView(R,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n"));

  /* C = A*R^T */
  CHKERRQ(MatMatTransposeMult(A,R,MAT_INITIAL_MATRIX,2.0,&C));
  CHKERRQ(MatSetOptionsPrefix(C,"ARt_"));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n"));

  /* Create MatTransposeColoring from symbolic C=A*R^T */
  CHKERRQ(MatColoringCreate(C,&mc));
  CHKERRQ(MatColoringSetDistance(mc,2));
  /* CHKERRQ(MatColoringSetType(mc,MATCOLORINGSL)); */
  CHKERRQ(MatColoringSetFromOptions(mc));
  CHKERRQ(MatColoringApply(mc,&iscoloring));
  CHKERRQ(MatColoringDestroy(&mc));
  CHKERRQ(MatTransposeColoringCreate(C,iscoloring,&matcoloring));
  CHKERRQ(ISColoringDestroy(&iscoloring));

  /* Create Rt_dense */
  CHKERRQ(MatCreate(PETSC_COMM_WORLD,&Rt_dense));
  CHKERRQ(MatSetSizes(Rt_dense,4,matcoloring->ncolors,PETSC_DECIDE,PETSC_DECIDE));
  CHKERRQ(MatSetType(Rt_dense,MATDENSE));
  CHKERRQ(MatSeqDenseSetPreallocation(Rt_dense,NULL));
  CHKERRQ(MatAssemblyBegin(Rt_dense,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(Rt_dense,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatGetLocalSize(Rt_dense,&m,&n));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"Rt_dense: %" PetscInt_FMT ",%" PetscInt_FMT "\n",m,n));

  /* Get Rt_dense by Apply MatTransposeColoring to R */
  CHKERRQ(MatTransColoringApplySpToDen(matcoloring,R,Rt_dense));

  /* C_dense = A*Rt_dense */
  CHKERRQ(MatMatMult(A,Rt_dense,MAT_INITIAL_MATRIX,2.0,&C_dense));
  CHKERRQ(MatSetOptionsPrefix(C_dense,"ARt_dense_"));
  /*CHKERRQ(MatView(C_dense,PETSC_VIEWER_STDOUT_WORLD)); */
  /*CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n")); */

  /* Recover C from C_dense */
  CHKERRQ(MatDuplicate(C,MAT_DO_NOT_COPY_VALUES,&C_sparse));
  CHKERRQ(MatTransColoringApplyDenToSp(matcoloring,C_dense,C_sparse));
  CHKERRQ(MatSetOptionsPrefix(C_sparse,"ARt_color_"));
  CHKERRQ(MatView(C_sparse,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"\n"));

  CHKERRQ(MatDestroy(&C_dense));
  CHKERRQ(MatDestroy(&C_sparse));
  CHKERRQ(MatDestroy(&Rt_dense));
  CHKERRQ(MatTransposeColoringDestroy(&matcoloring));
  CHKERRQ(MatDestroy(&C));

  /* Test PtAP = P^T*A*P, P = R^T */
  CHKERRQ(MatTranspose(R,MAT_INITIAL_MATRIX,&P));
  CHKERRQ(MatPtAP(A,P,MAT_INITIAL_MATRIX,2.0,&PtAP));
  CHKERRQ(MatSetOptionsPrefix(PtAP,"PtAP_"));
  CHKERRQ(MatView(PtAP,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&P));

  /* Test C = RARt */
  CHKERRQ(MatRARt(A,R,MAT_INITIAL_MATRIX,2.0,&C));
  CHKERRQ(MatRARt(A,R,MAT_REUSE_MATRIX,2.0,&C));
  CHKERRQ(MatEqual(C,PtAP,&equal));
  PetscCheck(equal,PETSC_COMM_SELF,PETSC_ERR_PLIB,"Error: PtAP != RARt");

  /* Free spaces */
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&R));
  CHKERRQ(MatDestroy(&PtAP));
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
