
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
  PetscErrorCode ierr;
  PetscBool      LU=PETSC_FALSE,CHOLESKY,TRIANGULAR=PETSC_FALSE,MATDSPL=PETSC_FALSE,flg,matordering;
  PetscScalar    v;
  IS             row,col;
  PetscViewer    viewer1,viewer2;
  MatFactorInfo  info;
  Vec            x,y,b,ytmp;
  PetscReal      norm2,norm2_inplace, tol = 100.*PETSC_MACHINE_EPSILON;
  PetscRandom    rdm;
  PetscMPIInt    size;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-lf",&lf,NULL);CHKERRQ(ierr);

  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,0,0,400,400,&viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,400,0,400,400,&viewer2);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m*n,m*n,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSetUp(C);CHKERRQ(ierr);

  /* Create matrix C in seqaij format and sC in seqsbaij. (This is five-point stencil with some extra elements) */
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      J = Ii - n; if (J>=0)  {ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      J = Ii + n; if (J<m*n) {ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      J = Ii - 1; if (J>=0)  {ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      J = Ii + 1; if (J<m*n) {ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatIsSymmetric(C,0.0,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_SELF,PETSC_ERR_SUP,"C is non-symmetric");

  /* Create vectors for error checking */
  ierr = MatCreateVecs(C,&x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&ytmp);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
  ierr = MatMult(C,x,b);CHKERRQ(ierr);

  ierr = PetscOptionsHasName(NULL,NULL,"-mat_ordering",&matordering);CHKERRQ(ierr);
  if (matordering) {
    ierr = MatGetOrdering(C,MATORDERINGRCM,&row,&col);CHKERRQ(ierr);
  } else {
    ierr = MatGetOrdering(C,MATORDERINGNATURAL,&row,&col);CHKERRQ(ierr);
  }

  ierr = PetscOptionsHasName(NULL,NULL,"-display_matrices",&MATDSPL);CHKERRQ(ierr);
  if (MATDSPL) {
    printf("original matrix:\n");
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = MatView(C,viewer1);CHKERRQ(ierr);
  }

  /* Compute LU or ILU factor A */
  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);

  info.fill          = 1.0;
  info.diagonal_fill = 0;
  info.zeropivot     = 0.0;

  ierr = PetscOptionsHasName(NULL,NULL,"-lu",&LU);CHKERRQ(ierr);
  if (LU) {
    printf("Test LU...\n");
    ierr = MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_LU,&A);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(A,C,row,col,&info);CHKERRQ(ierr);
  } else {
    printf("Test ILU...\n");
    info.levels = lf;

    ierr = MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_ILU,&A);CHKERRQ(ierr);
    ierr = MatILUFactorSymbolic(A,C,row,col,&info);CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric(A,C,&info);CHKERRQ(ierr);

  /* Solve A*y = b, then check the error */
  ierr = MatSolve(A,b,y);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);

  /* Test in-place ILU(0) and compare it with the out-place ILU(0) */
  if (!LU && lf==0) {
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&A);CHKERRQ(ierr);
    ierr = MatILUFactor(A,row,col,&info);CHKERRQ(ierr);
    /*
    printf("In-place factored matrix:\n");
    ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    */
    ierr = MatSolve(A,b,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm2_inplace);CHKERRQ(ierr);
    if (PetscAbs(norm2 - norm2_inplace) > tol) SETERRQ2(PETSC_COMM_SELF,1,"ILU(0) %g and in-place ILU(0) %g give different residuals",(double)norm2,(double)norm2_inplace);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }

  /* Test Cholesky and ICC on seqaij matrix with matrix reordering on aij matrix C */
  CHOLESKY = LU;
  if (CHOLESKY) {
    printf("Test Cholesky...\n");
    lf   = -1;
    ierr = MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_CHOLESKY,&A);CHKERRQ(ierr);
    ierr = MatCholeskyFactorSymbolic(A,C,row,&info);CHKERRQ(ierr);
  } else {
    printf("Test ICC...\n");
    info.levels        = lf;
    info.fill          = 1.0;
    info.diagonal_fill = 0;
    info.zeropivot     = 0.0;

    ierr = MatGetFactor(C,MATSOLVERPETSC,MAT_FACTOR_ICC,&A);CHKERRQ(ierr);
    ierr = MatICCFactorSymbolic(A,C,row,&info);CHKERRQ(ierr);
  }
  ierr = MatCholeskyFactorNumeric(A,C,&info);CHKERRQ(ierr);

  /* test MatForwardSolve() and MatBackwardSolve() with matrix reordering on aij matrix C */
  if (lf == -1) {
    ierr = PetscOptionsHasName(NULL,NULL,"-triangular_solve",&TRIANGULAR);CHKERRQ(ierr);
    if (TRIANGULAR) {
      printf("Test MatForwardSolve...\n");
      ierr = MatForwardSolve(A,b,ytmp);CHKERRQ(ierr);
      printf("Test MatBackwardSolve...\n");
      ierr = MatBackwardSolve(A,ytmp,y);CHKERRQ(ierr);
      ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
      ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
      if (norm2 > tol) {
        ierr = PetscPrintf(PETSC_COMM_SELF,"MatForwardSolve and BackwardSolve: Norm of error=%g\n",(double)norm2);CHKERRQ(ierr);
      }
    }
  }

  ierr = MatSolve(A,b,y);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
  if (lf == -1 && norm2 > tol) {
    PetscPrintf(PETSC_COMM_SELF, " reordered SEQAIJ:   Cholesky/ICC levels %d, residual %g\n",lf,norm2);CHKERRQ(ierr);
  }

  /* Test in-place ICC(0) and compare it with the out-place ICC(0) */
  if (!CHOLESKY && lf==0 && !matordering) {
    ierr = MatConvert(C,MATSBAIJ,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);
    ierr = MatICCFactor(A,row,&info);CHKERRQ(ierr);
    /*
    printf("In-place factored matrix:\n");
    ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    */
    ierr = MatSolve(A,b,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm2_inplace);CHKERRQ(ierr);
    if (PetscAbs(norm2 - norm2_inplace) > tol) SETERRQ2(PETSC_COMM_SELF,1,"ICC(0) %g and in-place ICC(0) %g give different residuals",(double)norm2,(double)norm2_inplace);
    ierr = MatDestroy(&A);CHKERRQ(ierr);
  }

  /* Free data structures */
  ierr = ISDestroy(&row);CHKERRQ(ierr);
  ierr = ISDestroy(&col);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&viewer2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(&rdm);CHKERRQ(ierr);
  ierr = VecDestroy(&x);CHKERRQ(ierr);
  ierr = VecDestroy(&y);CHKERRQ(ierr);
  ierr = VecDestroy(&ytmp);CHKERRQ(ierr);
  ierr = VecDestroy(&b);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
