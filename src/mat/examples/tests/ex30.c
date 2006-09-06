
static char help[] = "Tests ILU factorization and illustrates drawing of matrix sparsity structure with MatView().\n\
  Input parameters are:\n\
  -lf <level> : level of fill for ILU (default is 0)\n\
  -lu : use full LU factorization\n\
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
  ierr = PetscOptionsHasName(PETSC_NULL,"-lu",&LU);CHKERRQ(ierr);
  if (LU){ 
    ierr = MatLUFactorSymbolic(C,row,col,PETSC_NULL,&A);CHKERRQ(ierr);
  } else {
    ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
    info.levels        = lf;
    info.fill          = 1.0;
    info.diagonal_fill = 0;
    info.shiftnz       = 0;
    info.zeropivot     = 0.0;
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

  /* Test in-place ILU(0) */
  if (!LU){
    ierr = MatILUFactor(C,row,col,&info);CHKERRQ(ierr);
    /*
    printf("In-place factored matrix:\n");
    ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
    */  
    ierr = MatSolve(C,b,y);CHKERRQ(ierr);
    ierr = VecAXPY(y,-1.0,x);CHKERRQ(ierr);
    ierr = VecNorm(y,NORM_2,&norm2_inplace);CHKERRQ(ierr);
    if (PetscAbs(norm2 - norm2_inplace) > 1.e-16) SETERRQ2(1,"ILU(0) %G and in-place ILU(0) %G give different errors",norm2,norm2_inplace);
  }
  /* Free data structures */
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
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
