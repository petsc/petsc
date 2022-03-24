
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

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
  PetscCheckFalse(size != 1,PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-lf",&lf,NULL));

  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,0,0,400,400,&viewer1));
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,400,0,400,400,&viewer2));

  CHKERRQ(MatCreate(PETSC_COMM_SELF,&C));
  CHKERRQ(MatSetSizes(C,m*n,m*n,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  /* Create matrix C in seqaij format and sC in seqsbaij. (This is five-point stencil with some extra elements) */
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      J = Ii - n; if (J>=0)  CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii + n; if (J<m*n) CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii - 1; if (J>=0)  CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      J = Ii + 1; if (J<m*n) CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));
      v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatIsSymmetric(C,0.0,&flg));
  PetscCheck(flg,PETSC_COMM_SELF,PETSC_ERR_SUP,"C is non-symmetric");

  /* Create vectors for error checking */
  CHKERRQ(MatCreateVecs(C,&x,&b));
  CHKERRQ(VecDuplicate(x,&y));
  CHKERRQ(VecDuplicate(x,&ytmp));
  CHKERRQ(PetscRandomCreate(PETSC_COMM_SELF,&rdm));
  CHKERRQ(PetscRandomSetFromOptions(rdm));
  CHKERRQ(VecSetRandom(x,rdm));
  CHKERRQ(MatMult(C,x,b));

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-mat_ordering",&matordering));
  if (matordering) {
    CHKERRQ(MatGetOrdering(C,MATORDERINGRCM,&row,&col));
  } else {
    CHKERRQ(MatGetOrdering(C,MATORDERINGNATURAL,&row,&col));
  }

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-display_matrices",&MATDSPL));
  if (MATDSPL) {
    printf("original matrix:\n");
    CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_INFO));
    CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_SELF));
    CHKERRQ(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_SELF));
    CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_SELF));
    CHKERRQ(MatView(C,viewer1));
  }

  /* Compute LU or ILU factor A */
  CHKERRQ(MatFactorInfoInitialize(&info));

  info.fill          = 1.0;
  info.diagonal_fill = 0;
  info.zeropivot     = 0.0;

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-lu",&LU));
  if (LU) {
    printf("Test LU...\n");
    CHKERRQ(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&A));
    CHKERRQ(MatLUFactorSymbolic(A,C,row,col,&info));
  } else {
    printf("Test ILU...\n");
    info.levels = lf;

    CHKERRQ(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_ILU,&A));
    CHKERRQ(MatILUFactorSymbolic(A,C,row,col,&info));
  }
  CHKERRQ(MatLUFactorNumeric(A,C,&info));

  /* Solve A*y = b, then check the error */
  CHKERRQ(MatSolve(A,b,y));
  CHKERRQ(VecAXPY(y,-1.0,x));
  CHKERRQ(VecNorm(y,NORM_2,&norm2));
  CHKERRQ(MatDestroy(&A));

  /* Test in-place ILU(0) and compare it with the out-place ILU(0) */
  if (!LU && lf==0) {
    CHKERRQ(MatDuplicate(C,MAT_COPY_VALUES,&A));
    CHKERRQ(MatILUFactor(A,row,col,&info));
    /*
    printf("In-place factored matrix:\n");
    CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_SELF));
    */
    CHKERRQ(MatSolve(A,b,y));
    CHKERRQ(VecAXPY(y,-1.0,x));
    CHKERRQ(VecNorm(y,NORM_2,&norm2_inplace));
    PetscCheckFalse(PetscAbs(norm2 - norm2_inplace) > tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ILU(0) %g and in-place ILU(0) %g give different residuals",(double)norm2,(double)norm2_inplace);
    CHKERRQ(MatDestroy(&A));
  }

  /* Test Cholesky and ICC on seqaij matrix with matrix reordering on aij matrix C */
  CHOLESKY = LU;
  if (CHOLESKY) {
    printf("Test Cholesky...\n");
    lf   = -1;
    CHKERRQ(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&A));
    CHKERRQ(MatCholeskyFactorSymbolic(A,C,row,&info));
  } else {
    printf("Test ICC...\n");
    info.levels        = lf;
    info.fill          = 1.0;
    info.diagonal_fill = 0;
    info.zeropivot     = 0.0;

    CHKERRQ(MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_ICC,&A));
    CHKERRQ(MatICCFactorSymbolic(A,C,row,&info));
  }
  CHKERRQ(MatCholeskyFactorNumeric(A,C,&info));

  /* test MatForwardSolve() and MatBackwardSolve() with matrix reordering on aij matrix C */
  if (lf == -1) {
    CHKERRQ(PetscOptionsHasName(NULL,NULL,"-triangular_solve",&TRIANGULAR));
    if (TRIANGULAR) {
      printf("Test MatForwardSolve...\n");
      CHKERRQ(MatForwardSolve(A,b,ytmp));
      printf("Test MatBackwardSolve...\n");
      CHKERRQ(MatBackwardSolve(A,ytmp,y));
      CHKERRQ(VecAXPY(y,-1.0,x));
      CHKERRQ(VecNorm(y,NORM_2,&norm2));
      if (norm2 > tol) {
        CHKERRQ(PetscPrintf(PETSC_COMM_SELF,"MatForwardSolve and BackwardSolve: Norm of error=%g\n",(double)norm2));
      }
    }
  }

  CHKERRQ(MatSolve(A,b,y));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(VecAXPY(y,-1.0,x));
  CHKERRQ(VecNorm(y,NORM_2,&norm2));
  if (lf == -1 && norm2 > tol) {
    CHKERRQ(PetscPrintf(PETSC_COMM_SELF, " reordered SEQAIJ:   Cholesky/ICC levels %" PetscInt_FMT ", residual %g\n",lf,(double)norm2));
  }

  /* Test in-place ICC(0) and compare it with the out-place ICC(0) */
  if (!CHOLESKY && lf==0 && !matordering) {
    CHKERRQ(MatConvert(C,MATSBAIJ,MAT_INITIAL_MATRIX,&A));
    CHKERRQ(MatICCFactor(A,row,&info));
    /*
    printf("In-place factored matrix:\n");
    CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_SELF));
    */
    CHKERRQ(MatSolve(A,b,y));
    CHKERRQ(VecAXPY(y,-1.0,x));
    CHKERRQ(VecNorm(y,NORM_2,&norm2_inplace));
    PetscCheckFalse(PetscAbs(norm2 - norm2_inplace) > tol,PETSC_COMM_SELF,PETSC_ERR_PLIB,"ICC(0) %g and in-place ICC(0) %g give different residuals",(double)norm2,(double)norm2_inplace);
    CHKERRQ(MatDestroy(&A));
  }

  /* Free data structures */
  CHKERRQ(ISDestroy(&row));
  CHKERRQ(ISDestroy(&col));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscViewerDestroy(&viewer1));
  CHKERRQ(PetscViewerDestroy(&viewer2));
  CHKERRQ(PetscRandomDestroy(&rdm));
  CHKERRQ(VecDestroy(&x));
  CHKERRQ(VecDestroy(&y));
  CHKERRQ(VecDestroy(&ytmp));
  CHKERRQ(VecDestroy(&b));
  CHKERRQ(PetscFinalize());
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
