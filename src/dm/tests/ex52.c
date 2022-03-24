
static char help[] = "Tests periodic boundary conditions for DMDA1d with periodic boundary conditions.\n\n";

#include <petscdmda.h>

int main(int argc,char **argv)
{
  DM               da;
  Mat              A;
  const PetscInt   dfill[4] = {0,1,0,1},ofill[4] = {0,1,1,0};

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,7,2,1,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMDASetBlockFills(da,dfill,ofill));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateMatrix(da,&A));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  CHKERRQ(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(MatDestroy(&A));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2

TEST*/
