
static char help[] = "Tests PetscContainerCreate() and PetscContainerDestroy().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscContainer container;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscContainerCreate(PETSC_COMM_SELF,&container));
  CHKERRQ(PetscContainerDestroy(&container));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
