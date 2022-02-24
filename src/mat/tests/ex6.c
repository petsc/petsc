
static char help[] = "Tests reordering a matrix.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscInt       i,j,m = 5,n = 5,Ii,J;
  PetscErrorCode ierr;
  PetscScalar    v;
  IS             perm,iperm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));

  /* create the matrix for the five point stencil, YET AGAIN*/
  for (i=0; i<m; i++) {
    for (j=0; j<n; j++) {
      v = -1.0;  Ii = j + n*i;
      if (i>0)   {J = Ii - n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (i<m-1) {J = Ii + n; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j>0)   {J = Ii - 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      if (j<n-1) {J = Ii + 1; CHKERRQ(MatSetValues(C,1,&Ii,1,&J,&v,INSERT_VALUES));}
      v = 4.0; CHKERRQ(MatSetValues(C,1,&Ii,1,&Ii,&v,INSERT_VALUES));
    }
  }
  CHKERRQ(MatAssemblyBegin(C,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(C,MAT_FINAL_ASSEMBLY));

  CHKERRQ(MatGetOrdering(C,MATORDERINGND,&perm,&iperm));
  CHKERRQ(ISView(perm,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(ISView(iperm,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(ISDestroy(&iperm));
  CHKERRQ(MatDestroy(&C));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
