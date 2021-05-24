
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
  ierr = PetscOptionsHasName(NULL,NULL,"-baij",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = MatCreateSeqBAIJ(PETSC_COMM_WORLD,1,3,3,0,rowlens,&A);CHKERRQ(ierr);
    ierr = MatSeqBAIJSetColumnIndices(A,cols);CHKERRQ(ierr);
  } else {
    ierr = MatCreateSeqAIJ(PETSC_COMM_WORLD,3,3,0,rowlens,&A);CHKERRQ(ierr);
    ierr = MatSeqAIJSetColumnIndices(A,cols);CHKERRQ(ierr);
  }

  i    = 0; j = 0; v = 1.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  i    = 0; j = 2; v = 3.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

  i    = 1; j = 0; v = 1.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  i    = 1; j = 1; v = 2.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);
  i    = 1; j = 2; v = 3.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

  i    = 2; j = 2; v = 3.0;
  ierr = MatSetValues(A,1,&i,1,&j,&v,INSERT_VALUES);CHKERRQ(ierr);

  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatView(A,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      args: -baij

TEST*/
