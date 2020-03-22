
static char help[] = "Tests copying and ordering uniprocessor row-based sparse matrices.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C,A;
  PetscInt       i,j,m = 5,n = 5,Ii,J;
  PetscErrorCode ierr;
  PetscScalar    v;
  IS             perm,iperm;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,NULL,&C);CHKERRQ(ierr);
  /* create the matrix for the five point stencil, YET AGAIN*/
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

  ierr = MatConvert(C,MATSAME,MAT_INITIAL_MATRIX,&A);CHKERRQ(ierr);

  ierr = MatGetOrdering(A,MATORDERINGND,&perm,&iperm);CHKERRQ(ierr);
  ierr = ISView(perm,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = ISView(iperm,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF);CHKERRQ(ierr);

  ierr = ISDestroy(&perm);CHKERRQ(ierr);
  ierr = ISDestroy(&iperm);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:
      filter: grep -v "MPI processes"

TEST*/
