
static char help[] = "Tests ContainerCreate and ContainerDestroy.\n\n";

#include "petsc.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode       ierr;
  PetscObjectContainer container;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscObjectContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscObjectContainerDestroy(container);CHKERRQ(ierr); 
  ierr = PetscFinalize();CHKERRQ(ierr);
  return 0;
}
