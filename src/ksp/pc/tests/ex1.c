
static char help[] = "Tests the creation of a PC context.\n\n";

#include <petscpc.h>

int main(int argc,char **args)
{
  PC             pc;
  PetscErrorCode ierr;
  PetscInt       n = 5;
  Mat            mat;

  ierr = PetscInitialize(&argc,&args,(char*)0,help);if (ierr) return ierr;
  ierr = PCCreate(PETSC_COMM_WORLD,&pc);CHKERRQ(ierr);
  ierr = PCSetType(pc,PCNONE);CHKERRQ(ierr);

  /* Vector and matrix must be set before calling PCSetUp */
  ierr = MatCreateSeqAIJ(PETSC_COMM_SELF,n,n,3,NULL,&mat);CHKERRQ(ierr);
  ierr = MatAssemblyBegin(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(mat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PCSetOperators(pc,mat,mat);CHKERRQ(ierr);
  ierr = PCSetUp(pc);CHKERRQ(ierr);
  ierr = MatDestroy(&mat);CHKERRQ(ierr);
  ierr = PCDestroy(&pc);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}





/*TEST

   test:

TEST*/
