
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
  PetscInt       i,j,m = 5,n = 5,I,J,lf = 0;
  PetscErrorCode ierr;
  PetscTruth     flg1;
  PetscScalar    v;
  IS             row,col;
  PetscViewer    viewer1,viewer2;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-m",&m,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-lf",&lf,PETSC_NULL);CHKERRQ(ierr);

  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,0,0,400,400,&viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDrawOpen(PETSC_COMM_SELF,0,0,400,0,400,400,&viewer2);CHKERRQ(ierr);

  ierr = MatCreate(PETSC_COMM_SELF,m*n,m*n,m*n,m*n,&C);CHKERRQ(ierr);
  ierr = MatSetFromOptions(C);CHKERRQ(ierr);
  ierr = MatSeqBDiagSetPreallocation(C,0,1,PETSC_NULL,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatSeqAIJSetPreallocation(C,5,PETSC_NULL);CHKERRQ(ierr);

  /* Create the matrix. (This is five-point stencil with some extra elements) */
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  I = j + n*i;
      J = I - n; if (J>=0)  {ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      J = I + n; if (J<m*n) {ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      J = I - 1; if (J>=0)  {ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      J = I + 1; if (J<m*n) {ierr = MatSetValues(C,1,&I,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&I,1,&I,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatGetOrdering(C,MATORDERING_RCM,&row,&col);CHKERRQ(ierr);
  printf("original matrix:\n");
  ierr = PetscViewerPushFormat(PETSC_VIEWER_STDOUT_SELF,PETSC_VIEWER_ASCII_INFO);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = PetscViewerPopFormat(PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatView(C,viewer1);CHKERRQ(ierr);

  /* Compute factorization */
  ierr = PetscOptionsHasName(PETSC_NULL,"-lu",&flg1);CHKERRQ(ierr);
  if (flg1){ 
    ierr = MatLUFactorSymbolic(C,row,col,PETSC_NULL,&A);CHKERRQ(ierr);
  } else {
    MatFactorInfo info;

    ierr = MatFactorInfoInitialize(&info);CHKERRQ(ierr);
    info.levels        = lf;
    info.fill          = 1.0;
    info.diagonal_fill = 0;
    info.damping       = 0;
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

  /* Free data structures */
  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = MatDestroy(A);CHKERRQ(ierr);
  ierr = ISDestroy(row);CHKERRQ(ierr);
  ierr = ISDestroy(col);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer1);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(viewer2);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
