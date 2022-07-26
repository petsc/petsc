static char help[] = "Test basic DMProduct operations.\n\n";

#include <petscdm.h>
#include <petscdmproduct.h>

int main(int argc,char **argv)
{
  DM             dm;
  PetscInt       dim;

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  dim = 1;
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  PetscCall(DMCreate(PETSC_COMM_WORLD,&dm));
  PetscCall(DMSetType(dm,DMPRODUCT));
  PetscCall(DMSetDimension(dm,dim));
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMDestroy(&dm));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: basic_1

TEST*/
