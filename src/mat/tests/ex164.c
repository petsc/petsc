
static char help[] = "Tests MatConvert() from SeqDense to SeqAIJ \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,C;
  PetscInt       n = 10;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(MatCreateSeqDense(PETSC_COMM_WORLD,n,n,NULL,&A));
  CHKERRQ(MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&C));
  CHKERRQ(MatView(C,NULL));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(MatDestroy(&C));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
