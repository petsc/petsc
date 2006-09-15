
static char help[] = "Tests ILU and ICC factorization with matrix ordering, and illustrates drawing of matrix sparsity structure with MatView().\n\
  Input parameters are:\n\
  -lf <level> : level of fill for ILU (default is 0)\n\
  -lu : use full LU or Cholesky factorization\n\
  -m <value>,-n <value> : grid dimensions\n\
Note that most users should employ the KSP interface to the\n\
linear solvers instead of using the factorization routines\n\
directly.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C,A,sC,sA;; 
  PetscInt       i,j,m = 5,n = 5,Ii,J,lf = 0;
  PetscErrorCode ierr;
  PetscTruth     LU=PETSC_FALSE,flg;
  PetscScalar    v;
  IS             row,col;
  PetscViewer    viewer1,viewer2;
  MatFactorInfo  info;
  Vec            x,y,b;
  PetscReal      norm2,norm2_inplace;
  PetscRandom    rdm;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-lf",&lf,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,0,0,400,400,&viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,400,0,400,400,&viewer2);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m*n,m*n,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSeqBDiagSetPreallocation(C,0,1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(C,5,PETSC_NULL);CHKERRQ(ierr);

  /* Create the matrix. (This is five-point stencil with some extra elements) */
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
  if (!flg) SETERRQ(1,"C is non-symmetric");

  /* Create vectors for error checking */
  ierr = MatGetVecs(C,&x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
  ierr = MatMult(C,x,b);CHKERRQ(ierr);

  ierr = MatGetOrdering(C,MATORDERING_RCM,&row,&col);CHKERRQ(ierr);
  /* replace row or col with natural ordering for testing */
  ierr = PetscOptionsHasName(PETSC_NULL,"-no_rowperm",&flg);CHKERRQ(ierr);
  if (flg){
    ierr = ISDestroy(row);CHKERRQ(ierr);
    PetscInt *ii;
    ierr = PetscMalloc(m*n*sizeof(PetscInt),&ii);CHKERRQ(ierr);
    for (i=0; i<m*n; i++) ii[i] = i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m*n,ii,&row);CHKERRQ(ierr);
    ierr = PetscFree(ii);CHKERRQ(ierr);
    ierr = ISSetIdentity(row);CHKERRQ(ierr);
    ierr = ISSetPermutation(row);CHKERRQ(ierr);
  }
  ierr = PetscOptionsHasName(PETSC_NULL,"-no_colperm",&flg);CHKERRQ(ierr);
  if (flg){
    ierr = ISDestroy(col);CHKERRQ(ierr);
    PetscInt *ii;
    ierr = PetscMalloc(m*n*sizeof(PetscInt),&ii);CHKERRQ(ierr);
    for (i=0; i<m*n; i++) ii[i] = i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m*n,ii,&col);CHKERRQ(ierr);
    ierr = PetscFree(ii);CHKERRQ(ierr);
    ierr = ISSetIdentity(col);CHKERRQ(ierr);
    ierr = ISSetPermutation(col);CHKERRQ(ierr);
  }

  printf("original matrix:\n");
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatView(C,viewer1);CHKERRQ(ierr);

  /* Compute factorization */
  ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
  info.fill          = 1.0;
  info.diagonal_fill = 0;
  info.shiftnz       = 0;
  info.zeropivot     = 0.0;
  ierr = PetscOptionsHasName(PETSC_NULL,"-lu",&LU);CHKERRQ(ierr);
  if (LU){ 
    ierr = MatLUFactorSymbolic(C,row,col,&info,&A);CHKERRQ(ierr);
  } else {
    info.levels = lf;
    ierr = MatILUFactorSymbolic(C,row,col,&info,&A);CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric(C,&info,&A);CHKERRQ(ierr);

  printf("factored matrix:\n");
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatView(A,viewer2);CHKERRQ(ierr);

  /* Solve A*y = b, then check the error */
  ierr = MatSolve(A,b,y);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);

  /* Test in-place ILU(0) and compare it with the out-place ILU(0) */
  if (!LU && lf==0){
    ierr = MatDuplicate(C,MAT_COPY_VALUES,&A);CHKERRQ(ierr);
    ierr = MatILUFactor(A,row,col,&info);CHKERRQ(ierr);
    /*
    printf("In-place factored matrix:\n");
    ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    */  
    ierr = MatSolve(A,b,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm2_inplace);CHKERRQ(ierr);
    if (PetscAbs(norm2 - norm2_inplace) > 1.e-16) SETERRQ2(1,"ILU(0) %G and in-place ILU(0) %G give different residuals",norm2,norm2_inplace);
    ierr = MatDestroy(A);CHKERRQ(ierr);
  }

  /* Test Cholesky and ICC on seqaij matrix without matrix reordering */
  ierr = ISDestroy(row);CHKERRQ(ierr);
  ierr = ISDestroy(col);CHKERRQ(ierr);
  ierr = MatGetOrdering(C,MATORDERING_NATURAL,&row,&col);CHKERRQ(ierr);
  if (LU){ 
    lf = -1;
    ierr = MatCholeskyFactorSymbolic(C,row,&info,&A);CHKERRQ(ierr);
  } else {
    info.levels        = lf;
    info.fill          = 1.0;
    info.diagonal_fill = 0;
    info.shiftnz       = 0;
    info.zeropivot     = 0.0;
    ierr = MatICCFactorSymbolic(C,row,&info,&A);CHKERRQ(ierr);
  }
  ierr = MatCholeskyFactorNumeric(C,&info,&A);CHKERRQ(ierr);
  /*
  ierr = MatSolve(A,b,y);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
  printf(" SEQAIJ:   Cholesky/ICC levels %d, residual %g\n",lf,norm2);CHKERRQ(ierr);
  */

  /* Test Cholesky and ICC on seqsbaij matrix without matrix reordering */
  ierr = MatConvert(C,MATSEQSBAIJ,MAT_INITIAL_MATRIX,&sC);CHKERRQ(ierr);
  if (LU){ 
    ierr = MatCholeskyFactorSymbolic(sC,row,&info,&sA);CHKERRQ(ierr);
  } else {
    ierr = MatICCFactorSymbolic(sC,row,&info,&sA);CHKERRQ(ierr);
  }
  ierr = MatCholeskyFactorNumeric(sC,&info,&sA);CHKERRQ(ierr);
  /*
  ierr = MatSolve(sA,b,y);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
  printf(" SEQSBAIJ: Cholesky/ICC levels %d, residual %g\n",lf,norm2);CHKERRQ(ierr);
  */
  ierr = MatEqual(A,sA,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(1,"CholeskyFactors for aij and sbaij matrices are different");
  ierr = MatDestroy(sC);CHKERRQ(ierr);
  ierr = MatDestroy(sA);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = ISDestroy(row);CHKERRQ(ierr);
  ierr = ISDestroy(col);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rdm);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
