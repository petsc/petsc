
static char help[] = "Tests PetscContainerCreate() and PetscContainerDestroy().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscContainer container;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF,&container));
  CHKERRQ(PetscContainerDestroy(&container));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
