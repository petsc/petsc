static char help[] = "Test basic DMProduct operations.\n\n";

#include <petscdm.h>
#include <petscdmproduct.h>

int main(int argc,char **argv)
{
  DM             dm;
  PetscInt       dim;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  dim = 1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD,&dm));
  CHKERRQ(DMSetType(dm,DMPRODUCT));
  CHKERRQ(DMSetDimension(dm,dim));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(DMDestroy(&dm));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      suffix: basic_1

TEST*/
