
static char help[] = "Tests ILU and ICC factorization with and without matrix ordering on seqaij format, and illustrates drawing of matrix sparsity structure with MatView().\n\
  Input parameters are:\n\
  -lf <level> : level of fill for ILU (default is 0)\n\
  -lu : use full LU or Cholesky factorization\n\
  -m <value>,-n <value> : grid dimensions\n\
Note that most users should employ the KSP interface to the\n\
linear solvers instead of using the factorization routines\n\
directly.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i,j,m = 5,n = 5,Ii,J,lf = 0;
  PetscBool      LU=PETSC_FALSE,CHOLESKY,TRIANGULAR=PETSC_FALSE,MATDSPL=PETSC_FALSE,flg,matordering;
  PetscScalar    v;
  IS             row,col;
  PetscViewer    viewer1,viewer2;
  MatFactorInfo  info;
  Vec            x,y,b,ytmp;
  PetscReal      norm2,norm2_inplace, tol = 100.*PETSC_MACHINE_EPSILON;
  PetscRandom    rdm;
  PetscMPIInt    size;

  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheck(size == 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-lf",&lf,NULL));

  PetscCall(PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,0,0,400,400,&viewer1));
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,400,0,400,400,&viewer2));

  PetscCall(MatCreate(PETSC_COMM_SELF,&C));
  PetscCall(MatSetSizes(C,m*n,m*n,m*n,m*n));
  PetscCall(MatSetFromOptions(C));
  PetscCall(MatSetUp(C));

  /* Create matrix C in seqaij format and sC in seqsbaij. (This is five-point stencil with some extra elements) */
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      J = Ii - n; if (J>=0)  PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii + n; if (J<m*n) PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii - 1; if (J>=0)  PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii + 1; if (J<m*n) PetscCall(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      v = 4.0; PetscCall(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  PetscCall(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  PetscCall(MatIsSymmetric(C,0.0,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"C is non-symmetric");

  /* Create vectors for error checking */
  PetscCall(MatCreateVecs(C,&x,&b));
  PetscCall(VecDuplicate(x,&y));
  PetscCall(VecDuplicate(x,&ytmp));
  PetscCall(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  PetscCall(PetscRandomSetFromOptions(rdm));
  PetscCall(VecSetRandom(x,rdm));
  PetscCall(MatMult(C,x,b));

  PetscCall(PetscOptionsHasName(NULL,NULL,"-mat_ordering",&matordering));
  if (matordering) {
    PetscCall(MatGetOrdering(C,MATORDERINGRCM,&row,&col));
  } else {
    PetscCall(MatGetOrdering(C,MATORDERINGNATURAL,&row,&col));
  }

  PetscCall(PetscOptionsHasName(NULL,NULL,"-display_matrices",&MATDSPL));
  if (MATDSPL) {
    printf("original matrix:\n");
    PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_INFO));
    PetscCall(MatView(C,PETSC_VIEWER_STDOUT_SELF));
    PetscCall(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_SELF));
    PetscCall(MatView(C,PETSC_VIEWER_STDOUT_SELF));
    PetscCall(MatView(C,viewer1));
  }

  /* Compute LU or ILU factor A */
  PetscCall(MatFactorInfoInitialize(&info));

  info.fill          = 1.0;
  info.diagonal_fill = 0;
  info.zeropivot     = 0.0;

  PetscCall(PetscOptionsHasName(NULL,NULL,"-lu",&LU));
  if (LU) {
    printf("Test LU...\n");
    PetscCall(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&A));
    PetscCall(MatLUFactorSymbolic(A,C,row,col,&info));
  } else {
    printf("Test ILU...\n");
    info.levels = lf;

    PetscCall(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_ILU,&A));
    PetscCall(MatILUFactorSymbolic(A,C,row,col,&info));
  }
  PetscCall(MatLUFactorNumeric(A,C,&info));

  /* Solve A*y = b, then check the error */
  PetscCall(MatSolve(A,b,y));
  PetscCall(VecAXPY(y,-1.0,x));
  PetscCall(VecNorm(y,NORM_2,&norm2));
  PetscCall(MatDestroy(&A));

  /* Test in-place ILU(0) and compare it with the out-place ILU(0) */
  if (!LU && lf==0) {
    PetscCall(MatDuplicate(C,MAT_COPY_VALUES,&A));
    PetscCall(MatILUFactor(A,row,col,&info));
    /*
    printf("In-place factored matrix:\n");
    PetscCall(MatView(C,PETSC_VIEWER_STDOUT_SELF));
    */
    PetscCall(MatSolve(A,b,y));
    PetscCall(VecAXPY(y,-1.0,x));
    PetscCall(VecNorm(y,NORM_2,&norm2_inplace));
    PetscCheck(PetscAbs(norm2 - norm2_inplace) <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ILU(0) %g and in-place ILU(0) %g give different residuals",(double)norm2,(double)norm2_inplace);
    PetscCall(MatDestroy(&A));
  }

  /* Test Cholesky and ICC on seqaij matrix with matrix reordering on aij matrix C */
  CHOLESKY = LU;
  if (CHOLESKY) {
    printf("Test Cholesky...\n");
    lf   = -1;
    PetscCall(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&A));
    PetscCall(MatCholeskyFactorSymbolic(A,C,row,&info));
  } else {
    printf("Test ICC...\n");
    info.levels        = lf;
    info.fill          = 1.0;
    info.diagonal_fill = 0;
    info.zeropivot     = 0.0;

    PetscCall(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_ICC,&A));
    PetscCall(MatICCFactorSymbolic(A,C,row,&info));
  }
  PetscCall(MatCholeskyFactorNumeric(A,C,&info));

  /* test MatForwardSolve() and MatBackwardSolve() with matrix reordering on aij matrix C */
  if (lf == -1) {
    PetscCall(PetscOptionsHasName(NULL,NULL,"-triangular_solve",&TRIANGULAR));
    if (TRIANGULAR) {
      printf("Test MatForwardSolve...\n");
      PetscCall(MatForwardSolve(A,b,ytmp));
      printf("Test MatBackwardSolve...\n");
      PetscCall(MatBackwardSolve(A,ytmp,y));
      PetscCall(VecAXPY(y,-1.0,x));
      PetscCall(VecNorm(y,NORM_2,&norm2));
      if (norm2 > tol) {
        PetscCall(PetscPrintf(PETSC_COMM_SELF,"MatForwardSolve and BackwardSolve: Norm of error=%g\n",(double)norm2));
      }
    }
  }

  PetscCall(MatSolve(A,b,y));
  PetscCall(MatDestroy(&A));
  PetscCall(VecAXPY(y,-1.0,x));
  PetscCall(VecNorm(y,NORM_2,&norm2));
  if (lf == -1 && norm2 > tol) {
    PetscCall(PetscPrintf(PETSC_COMM_SELF, " reordered SEQAIJ:   Cholesky/ICC levels %" PetscInt_FMT ", residual %g\n",lf,(double)norm2));
  }

  /* Test in-place ICC(0) and compare it with the out-place ICC(0) */
  if (!CHOLESKY && lf==0 && !matordering) {
    PetscCall(MatConvert(C,MATSBAIJ,MAT_INITIAL_MATRIX,&A));
    PetscCall(MatICCFactor(A,row,&info));
    /*
    printf("In-place factored matrix:\n");
    PetscCall(MatView(A,PETSC_VIEWER_STDOUT_SELF));
    */
    PetscCall(MatSolve(A,b,y));
    PetscCall(VecAXPY(y,-1.0,x));
    PetscCall(VecNorm(y,NORM_2,&norm2_inplace));
    PetscCheck(PetscAbs(norm2 - norm2_inplace) <= tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ICC(0) %g and in-place ICC(0) %g give different residuals",(double)norm2,(double)norm2_inplace);
    PetscCall(MatDestroy(&A));
  }

  /* Free data structures */
  PetscCall(ISDestroy(&row));
  PetscCall(ISDestroy(&col));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscViewerDestroy(&viewer1));
  PetscCall(PetscViewerDestroy(&viewer2));
  PetscCall(PetscRandomDestroy(&rdm));
  PetscCall(VecDestroy(&x));
  PetscCall(VecDestroy(&y));
  PetscCall(VecDestroy(&ytmp));
  PetscCall(VecDestroy(&b));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -mat_ordering -display_matrices -nox
      filter: grep -v "MPI processes"

   test:
      suffix: 2
      args: -mat_ordering -display_matrices -nox -lu

   test:
      suffix: 3
      args: -mat_ordering -lu -triangular_solve

   test:
      suffix: 4

   test:
      suffix: 5
      args: -lu

   test:
      suffix: 6
      args: -lu -triangular_solve
      output_file: output/ex30_3.out

TEST*/
