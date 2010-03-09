
static char help[] = "Tests ILU and ICC factorization with and without matrix ordering on seqaij format, and illustrates drawing of matrix sparsity structure with MatView().\n\
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
  Mat            C,A;
  PetscInt       i,j,m = 5,n = 5,Ii,J,lf = 0;
  PetscErrorCode ierr;
  PetscTruth     LU=PETSC_FALSE,CHOLESKY,TRIANGULAR=PETSC_FALSE,MATDSPL=PETSC_FALSE,flg;
  PetscScalar    v;
  IS             row,col;
  PetscViewer    viewer1,viewer2;
  MatFactorInfo  info;
  Vec            x,y,b,ytmp;
  PetscReal      norm2,norm2_inplace;
  PetscRandom    rdm;
  PetscInt       *ii;
  PetscMPIInt    size;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRQ(ierr);
  if (size != 1) SETERRQ(PETSC_ERR_SUP,"This is a uniprocessor example only!");
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-lf",&lf,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,0,0,400,400,&viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,400,0,400,400,&viewer2);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,&C);CHKERRQ(ierr);
  ierr = MatSetSizes(C,m*n,m*n,m*n,m*n);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);

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
  if (!flg) SETERRQ(1,"C is non-symmetric");

  /* Create vectors for error checking */
  ierr = MatGetVecs(C,&x,&b);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&y);CHKERRQ(ierr);
  ierr = VecDuplicate(x,&ytmp);CHKERRQ(ierr);
  ierr = PetscRandomCreate(PETSC_COMM_SELF,&rdm);CHKERRQ(ierr);
  ierr = PetscRandomSetFromOptions(rdm);CHKERRQ(ierr);
  ierr = VecSetRandom(x,rdm);CHKERRQ(ierr);
  ierr = MatMult(C,x,b);CHKERRQ(ierr);

  ierr = MatGetOrdering(C,MATORDERING_RCM,&row,&col);CHKERRQ(ierr);
  /* replace row or col with natural ordering for testing */
  ierr = PetscOptionsHasName(PETSC_NULL,"-no_rowperm",&flg);CHKERRQ(ierr);
  if (flg){
    ierr = ISDestroy(row);CHKERRQ(ierr);
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
    ierr = PetscMalloc(m*n*sizeof(PetscInt),&ii);CHKERRQ(ierr);
    for (i=0; i<m*n; i++) ii[i] = i;
    ierr = ISCreateGeneral(PETSC_COMM_SELF,m*n,ii,&col);CHKERRQ(ierr);
    ierr = PetscFree(ii);CHKERRQ(ierr);
    ierr = ISSetIdentity(col);CHKERRQ(ierr);
    ierr = ISSetPermutation(col);CHKERRQ(ierr);
  }

  ierr = PetscOptionsHasName(PETSC_NULL,"-display_matrices",&MATDSPL);CHKERRQ(ierr);
  if (MATDSPL){
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
  ierr = PetscOptionsHasName(PETSC_NULL,"-lu",&LU);CHKERRQ(ierr);
  if (LU){ 
    printf("Test LU...\n");
    ierr = MatGetFactor(C,MAT_SOLVER_PETSC,MAT_FACTOR_LU,&A);CHKERRQ(ierr);
    ierr = MatLUFactorSymbolic(A,C,row,col,&info);CHKERRQ(ierr);
  } else {
    printf("Test ILU...\n");
    info.levels = lf;
    ierr = MatGetFactor(C,MAT_SOLVER_PETSC,MAT_FACTOR_ILU,&A);CHKERRQ(ierr);
    ierr = MatILUFactorSymbolic(A,C,row,col,&info);CHKERRQ(ierr);
  }
  ierr = MatLUFactorNumeric(A,C,&info);CHKERRQ(ierr);

  if (MATDSPL){
    printf("factored matrix:\n");
    ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
    ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    ierr = MatView(A,viewer2);CHKERRQ(ierr);
  }

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

  /* Test Cholesky and ICC on seqaij matrix with matrix reordering on aij matrix C */
  CHOLESKY = LU;
  if (CHOLESKY){ 
    printf("Test Cholesky...\n");
    lf = -1;
    ierr = MatGetFactor(C,MAT_SOLVER_PETSC,MAT_FACTOR_CHOLESKY,&A);CHKERRQ(ierr);
    ierr = MatCholeskyFactorSymbolic(A,C,row,&info);CHKERRQ(ierr);
  } else {
    printf("Test ICC...\n");
    info.levels        = lf;
    info.fill          = 1.0;
    info.diagonal_fill = 0;
    info.zeropivot     = 0.0;
    ierr = MatGetFactor(C,MAT_SOLVER_PETSC,MAT_FACTOR_ICC,&A);CHKERRQ(ierr);
    ierr = MatICCFactorSymbolic(A,C,row,&info);CHKERRQ(ierr);
  }
  ierr = MatCholeskyFactorNumeric(A,C,&info);CHKERRQ(ierr);  

  /* test MatForwardSolve() and MatBackwardSolve() with matrix reordering on aij matrix C */
  if (lf == -1){
    ierr = PetscOptionsHasName(PETSC_NULL,"-triangular_solve",&TRIANGULAR);CHKERRQ(ierr);
    if (TRIANGULAR){
      printf("Test MatForwardSolve...\n");
      ierr = MatForwardSolve(A,b,ytmp);CHKERRQ(ierr);
      printf("Test MatBackwardSolve...\n");
      ierr = MatBackwardSolve(A,ytmp,y);CHKERRQ(ierr);      
      ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
      ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
      if (norm2 > 1.e-14){
        ierr = PetscPrintf(PETSC_COMM_SELF,"MatForwardSolve and BackwardSolve: Norm of error=%G\n",norm2);CHKERRQ(ierr); 
      }
    }
  } 

  ierr = MatSolve(A,b,y);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
  ierr = VecNorm(y,NORM_2,&norm2);CHKERRQ(ierr);
  if (lf == -1 && norm2 > 1.e-14){
    PetscPrintf(PETSC_COMM_SELF, " reordered SEQAIJ:   Cholesky/ICC levels %d, residual %g\n",lf,norm2);CHKERRQ(ierr);
  }
  ierr = ISDestroy(row);CHKERRQ(ierr);
  ierr = ISDestroy(col);CHKERRQ(ierr);

  /* Free data structures */
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer2);CHKERRQ(ierr);
  ierr = PetscRandomDestroy(rdm);CHKERRQ(ierr);
  ierr = VecDestroy(x);CHKERRQ(ierr);
  ierr = VecDestroy(y);CHKERRQ(ierr);
  ierr = VecDestroy(ytmp);CHKERRQ(ierr);
  ierr = VecDestroy(b);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
