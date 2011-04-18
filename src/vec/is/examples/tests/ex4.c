
static char help[] = "Tests ISToGeneral().\n\n";

#include <petscis.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  PetscInt        step = 2;
  IS              is;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  ierr = PetscOptionsGetInt(PETSC_NULL,"-step",&step,PETSC_NULL);CHKERRQ(ierr);
  ierr = ISCreateStride(PETSC_COMM_SELF,10,0,step,&is);CHKERRQ(ierr);

  ierr = ISToGeneral(is);CHKERRQ(ierr);

  ierr = ISDestroy(&is);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}
 






