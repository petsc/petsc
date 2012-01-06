
static char help[] = "Tests ContainerCreate and ContainerDestroy.\n\n";

#include <petscsys.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscContainer container;

  PetscInitialize(&argc,&argv,(char *)0,help);
  ierr = PetscContainerCreate(PETSC_COMM_SELF,&container);CHKERRQ(ierr);
  ierr = PetscContainerDestroy(container);CHKERRQ(ierr); 
  ierr = PetscFinalize();
  return 0;
}
