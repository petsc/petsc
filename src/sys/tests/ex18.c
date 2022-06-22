
static char help[] = "Tests PetscContainerCreate() and PetscContainerDestroy().\n\n";

#include <petscsys.h>

int main(int argc,char **argv)
{
  PetscContainer container;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscContainerCreate(PETSC_COMM_SELF,&container));
  PetscCall(PetscContainerDestroy(&container));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
