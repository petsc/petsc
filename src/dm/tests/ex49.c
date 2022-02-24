static char help[] = "Test basic DMProduct operations.\n\n";

#include <petscdm.h>
#include <petscdmproduct.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  DM             dm;
  PetscInt       dim;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  dim = 1;
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,NULL));
  CHKERRQ(DMCreate(PETSC_COMM_WORLD,&dm));
  CHKERRQ(DMSetType(dm,DMPRODUCT));
  CHKERRQ(DMSetDimension(dm,dim));
  CHKERRQ(DMSetFromOptions(dm));
  CHKERRQ(DMSetUp(dm));
  CHKERRQ(DMDestroy(&dm));
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: basic_1

TEST*/
