
static char help[] = "Tests MatConvert() from SeqDense to SeqAIJ \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,C;
  PetscErrorCode ierr;
  PetscInt       n = 10;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = MatCreateSeqDense(PETSC_COMM_WORLD,n,n,NULL,&A);CHKERRQ(ierr);
  ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatView(C,NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr);
  ierr = MatDestroy(&C);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}


/*TEST

   test:

TEST*/
