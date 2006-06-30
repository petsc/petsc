
static char help[] = "Tests matrix ordering routines.\n\n";

#include "petscmat.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            C; 
  PetscInt       i,j,m = 5,n = 5,Ii,J;
  PetscErrorCode ierr;
  PetscScalar    v;
  IS             perm,iperm;

  PetscInitialize(&argc,&args,(char *)0,help);

  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,PETSC_NULL,&C); 

  /* create the matrix for the five point stencil, YET AGAIN */
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (i<m-1) {J = Ii + n; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j>0)   {J = Ii - 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      if (j<n-1) {J = Ii + 1; ierr = MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES);CHKERRQ(ierr);}
      v = 4.0; ierr = MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES);CHKERRQ(ierr);
    }
  }
  ierr = MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);

  ierr = MatGetOrdering(C,MATORDERING_ND,&perm,&iperm);CHKERRQ(ierr);
  ierr = ISView(perm,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = ISDestroy(iperm);CHKERRQ(ierr);

  ierr = MatGetOrdering(C,MATORDERING_RCM,&perm,&iperm);CHKERRQ(ierr);
  ierr = ISView(perm,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = ISDestroy(iperm);CHKERRQ(ierr);

  ierr = MatGetOrdering(C,MATORDERING_QMD,&perm,&iperm);CHKERRQ(ierr);
  ierr = ISView(perm,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISDestroy(perm);CHKERRQ(ierr);
  ierr = ISDestroy(iperm);CHKERRQ(ierr);

  ierr = MatDestroy(C);CHKERRQ(ierr);
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
