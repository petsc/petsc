
static char help[] = "Tests MatSeq(B)AIJSetColumnIndices().\n\n";

#include <petscmat.h>

/*
      Generate the following matrix:

         1 0 3
         1 2 3
         0 0 3
*/
int main(int argc,char **args)
{
  Mat            A;
  PetscScalar    v;
  PetscErrorCode ierr;
  PetscInt       i,j,rowlens[] = {2,3,1},cols[] = {0,2,0,1,2,2};
  PetscBool      flg;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsHasName(NULL,NULL,"-baij",&flg));
  if (flg) {
    CHKERRQ(MatCreateSeqBAIJ(PETSC_COMM_WORLD,1,3,3,0,rowlens,&A));
    CHKERRQ(MatSeqBAIJSetColumnIndices(A,cols));
  } else {
    CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_WORLD,3,3,0,rowlens,&A));
    CHKERRQ(MatSeqAIJSetColumnIndices(A,cols));
  }

  i    = 0; j = 0; v = 1.0;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 0; j = 2; v = 3.0;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));

  i    = 1; j = 0; v = 1.0;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 1; j = 1; v = 2.0;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));
  i    = 1; j = 2; v = 3.0;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));

  i    = 2; j = 2; v = 3.0;
  CHKERRQ(MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES));

  CHKERRQ(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(MatDestroy(&A));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      args: -baij

TEST*/
