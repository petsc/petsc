
static char help[] = "Tests MatConvert() from SeqDense to SeqAIJ \n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
  Mat            A,C;
  PetscInt       n = 10;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(MatCreateSeqDense(PETSC_COMM_WORLD,n,n,NULL,&A));
  PetscCall(MatConvert(A,MATSEQDENSE,MAT_INITIAL_MATRIX,&C));
  PetscCall(MatView(C,NULL));
  PetscCall(MatDestroy(&A));
  PetscCall(MatDestroy(&C));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
