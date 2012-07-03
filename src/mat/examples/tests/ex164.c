 
static char help[] = "Tests MatConvert() from SeqDense to SeqAIJ \n\n";

#include <petscmat.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            A,C; 
  PetscErrorCode ierr;
  PetscInt       n = 10;

  PetscInitialize(&argc,&args,(char *)0,help);
  ierr = MatCreateSeqDense(PETSC_COMM_WORLD,n,n,PETSC_NULL,&A);CHKERRQ(ierr);
  ierr = MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&C);CHKERRQ(ierr);
  ierr = MatView(C,PETSC_NULL);CHKERRQ(ierr);
  ierr = MatDestroy(&A);CHKERRQ(ierr); 
  ierr = MatDestroy(&C);CHKERRQ(ierr); 
  ierr = PetscFinalize();
  return 0;
}
