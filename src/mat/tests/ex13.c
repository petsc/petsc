
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
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,m*n,m*n,5,NULL,&C));
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

  CHKERRQ(MatConvert(C,MATSAME,MAT_INITIAL_MATRIX,&A));

  CHKERRQ(MatGetOrdering(A,MATORDERINGND,&perm,&iperm));
  CHKERRQ(ISView(perm,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(ISView(iperm,PETSC_VIEWER_STDOUT_SELF));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(ISDestroy(&perm));
  CHKERRQ(ISDestroy(&iperm));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(MatDestroy(&A));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      filter: grep -v "MPI processes"

TEST*/
