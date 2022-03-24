
static char help[] = "Tests the use of MatZeroRows() for uniprocessor matrices.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            C;
  PetscInt       i,j,m = 5,n = 5,Ii,J;
  PetscScalar    v,five = 5.0;
  IS             isrow;
  PetscBool      keepnonzeropattern;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  /* create the matrix for the five point stencil, YET AGAIN*/
  CHKERRQ(MatCreate(PETSC_COMM_SELF,&C));
  CHKERRQ(MatSetSizes(C,PETSC_DECIDE,PETSC_DECIDE,m*n,m*n));
  CHKERRQ(MatSetFromOptions(C));
  CHKERRQ(MatSetUp(C));
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

  CHKERRQ(ISCreateStride(PETSC_COMM_SELF,(m*n)/2,0,2,&isrow));

  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-keep_nonzero_pattern",&keepnonzeropattern));
  if (keepnonzeropattern) {
    CHKERRQ(MatSetOption(C,MAT_KEEP_NONZERO_PATTERN,PETSC_TRUE));
  }

  CHKERRQ(MatZeroRowsIS(C,isrow,five,0,0));

  CHKERRQ(MatView(C,PETSC_VIEWER_STDOUT_SELF));

  CHKERRQ(ISDestroy(&isrow));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      args: -mat_type seqbaij -mat_block_size 5

   test:
      suffix: 3
      args: -keep_nonzero_pattern

   test:
      suffix: 4
      args: -keep_nonzero_pattern -mat_type seqbaij -mat_block_size 5

TEST*/
