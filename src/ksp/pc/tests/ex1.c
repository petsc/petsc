
static char help[] = "Tests the creation of a PC context.\n\n";

#include <petscpc.h>

int main(int argc,char **args)
{
  PC             pc;
  PetscInt       n = 5;
  Mat            mat;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&args,(char*)0,help));
  PetscCall(PCCreate(PETSC_COMM_WORLD,&pc));
  PetscCall(PCSetType(pc,PCNONE));

  /* Vector and matrix must be set before calling PCSetUp */
  PetscCall(MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,NULL,&mat));
  PetscCall(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  PetscCall(PCSetOperators(pc,mat,mat));
  PetscCall(PCSetUp(pc));
  PetscCall(MatDestroy(&mat));
  PetscCall(PCDestroy(&pc));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
