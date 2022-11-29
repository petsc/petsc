static char help[] = "Test MatTransposeColoring for SeqAIJ matrices. Used for '-matmattransmult_color' on  MatMatTransposeMult \n\n";

#include <petscmat.h>
#include <petsc/private/matimpl.h> /* Need struct _p_MatTransposeColoring for this test. */

int main(int argc, char **argv)
{
  Mat                  A, R, C, C_dense, C_sparse, Rt_dense, P, PtAP;
  PetscInt             row, col, m, n;
  MatScalar            one = 1.0, val;
  MatColoring          mc;
  MatTransposeColoring matcoloring = 0;
  ISColoring           iscoloring;
  PetscBool            equal;
  PetscMPIInt          size;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc, &argv, (char *)0, help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD, &size));
  PetscCheck(size == 1, PETSC_COMM_WORLD, PETSC_ERR_WRONG_MPI_SIZE, "This is a uniprocessor example only!");

  /* Create seqaij A */
  PetscCall(MatCreate(PETSC_COMM_SELF, &A));
  PetscCall(MatSetSizes(A, 4, 4, 4, 4));
  PetscCall(MatSetType(A, MATSEQAIJ));
  PetscCall(MatSetFromOptions(A));
  PetscCall(MatSetUp(A));
  row = 0;
  col = 0;
  val = 1.0;
  PetscCall(MatSetValues(A, 1, &row, 1, &col, &val, ADD_VALUES));
  row = 1;
  col = 3;
  val = 2.0;
  PetscCall(MatSetValues(A, 1, &row, 1, &col, &val, ADD_VALUES));
  row = 2;
  col = 2;
  val = 3.0;
  PetscCall(MatSetValues(A, 1, &row, 1, &col, &val, ADD_VALUES));
  row = 3;
  col = 0;
  val = 4.0;
  PetscCall(MatSetValues(A, 1, &row, 1, &col, &val, ADD_VALUES));
  PetscCall(MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOptionsPrefix(A, "A_"));
  PetscCall(MatView(A, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));

  /* Create seqaij R */
  PetscCall(MatCreate(PETSC_COMM_SELF, &R));
  PetscCall(MatSetSizes(R, 2, 4, 2, 4));
  PetscCall(MatSetType(R, MATSEQAIJ));
  PetscCall(MatSetFromOptions(R));
  PetscCall(MatSetUp(R));
  row = 0;
  col = 0;
  PetscCall(MatSetValues(R, 1, &row, 1, &col, &one, ADD_VALUES));
  row = 0;
  col = 1;
  PetscCall(MatSetValues(R, 1, &row, 1, &col, &one, ADD_VALUES));

  row = 1;
  col = 1;
  PetscCall(MatSetValues(R, 1, &row, 1, &col, &one, ADD_VALUES));
  row = 1;
  col = 2;
  PetscCall(MatSetValues(R, 1, &row, 1, &col, &one, ADD_VALUES));
  row = 1;
  col = 3;
  PetscCall(MatSetValues(R, 1, &row, 1, &col, &one, ADD_VALUES));
  PetscCall(MatAssemblyBegin(R, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(R, MAT_FINAL_ASSEMBLY));
  PetscCall(MatSetOptionsPrefix(R, "R_"));
  PetscCall(MatView(R, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));

  /* C = A*R^T */
  PetscCall(MatMatTransposeMult(A, R, MAT_INITIAL_MATRIX, 2.0, &C));
  PetscCall(MatSetOptionsPrefix(C, "ARt_"));
  PetscCall(MatView(C, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));

  /* Create MatTransposeColoring from symbolic C=A*R^T */
  PetscCall(MatColoringCreate(C, &mc));
  PetscCall(MatColoringSetDistance(mc, 2));
  /* PetscCall(MatColoringSetType(mc,MATCOLORINGSL)); */
  PetscCall(MatColoringSetFromOptions(mc));
  PetscCall(MatColoringApply(mc, &iscoloring));
  PetscCall(MatColoringDestroy(&mc));
  PetscCall(MatTransposeColoringCreate(C, iscoloring, &matcoloring));
  PetscCall(ISColoringDestroy(&iscoloring));

  /* Create Rt_dense */
  PetscCall(MatCreate(PETSC_COMM_WORLD, &Rt_dense));
  PetscCall(MatSetSizes(Rt_dense, 4, matcoloring->ncolors, PETSC_DECIDE, PETSC_DECIDE));
  PetscCall(MatSetType(Rt_dense, MATDENSE));
  PetscCall(MatSeqDenseSetPreallocation(Rt_dense, NULL));
  PetscCall(MatAssemblyBegin(Rt_dense, MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(Rt_dense, MAT_FINAL_ASSEMBLY));
  PetscCall(MatGetLocalSize(Rt_dense, &m, &n));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "Rt_dense: %" PetscInt_FMT ",%" PetscInt_FMT "\n", m, n));

  /* Get Rt_dense by Apply MatTransposeColoring to R */
  PetscCall(MatTransColoringApplySpToDen(matcoloring, R, Rt_dense));

  /* C_dense = A*Rt_dense */
  PetscCall(MatMatMult(A, Rt_dense, MAT_INITIAL_MATRIX, 2.0, &C_dense));
  PetscCall(MatSetOptionsPrefix(C_dense, "ARt_dense_"));
  /*PetscCall(MatView(C_dense,PETSC_VIEWER_STDOUT_WORLD)); */
  /*PetscCall(PetscPrintf(PETSC_COMM_SELF,"\n")); */

  /* Recover C from C_dense */
  PetscCall(MatDuplicate(C, MAT_DO_NOT_COPY_VALUES, &C_sparse));
  PetscCall(MatTransColoringApplyDenToSp(matcoloring, C_dense, C_sparse));
  PetscCall(MatSetOptionsPrefix(C_sparse, "ARt_color_"));
  PetscCall(MatView(C_sparse, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscPrintf(PETSC_COMM_SELF, "\n"));

  PetscCall(MatDestroy(&C_dense));
  PetscCall(MatDestroy(&C_sparse));
  PetscCall(MatDestroy(&Rt_dense));
  PetscCall(MatTransposeColoringDestroy(&matcoloring));
  PetscCall(MatDestroy(&C));

  /* Test PtAP = P^T*A*P, P = R^T */
  PetscCall(MatTranspose(R, MAT_INITIAL_MATRIX, &P));
  PetscCall(MatPtAP(A, P, MAT_INITIAL_MATRIX, 2.0, &PtAP));
  PetscCall(MatSetOptionsPrefix(PtAP, "PtAP_"));
  PetscCall(MatView(PtAP, PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&P));

  /* Test C = RARt */
  PetscCall(MatRARt(A, R, MAT_INITIAL_MATRIX, 2.0, &C));
  PetscCall(MatRARt(A, R, MAT_REUSE_MATRIX, 2.0, &C));
  PetscCall(MatEqual(C, PtAP, &equal));
  PetscCheck(equal, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Error: PtAP != RARt");

  /* Free spaces */
  PetscCall(MatDestroy(&C));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&R));
  PetscCall(MatDestroy(&PtAP));
  PetscCall(PetscFinalize());
  return 0;
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
