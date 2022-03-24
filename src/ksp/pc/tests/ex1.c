
static char help[] = "Tests the creation of a PC context.\n\n";

#include <petscpc.h>

int main(int argc,char **args)
{
  PC             pc;
  PetscInt       n = 5;
  Mat            mat;

  CHKERRQ(PetscInitialize(&argc,&args,(char*)0,help));
  CHKERRQ(PCCreate(PETSC_COMM_WORLD,&pc));
  CHKERRQ(PCSetType(pc,PCNONE));

  /* Vector and matrix must be set before calling PCSetUp */
  CHKERRQ(MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,NULL,&mat));
  CHKERRQ(MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY));
  CHKERRQ(PCSetOperators(pc,mat,mat));
  CHKERRQ(PCSetUp(pc));
  CHKERRQ(MatDestroy(&mat));
  CHKERRQ(PCDestroy(&pc));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
