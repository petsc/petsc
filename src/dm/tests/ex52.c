
static char help[] = "Tests periodic boundary conditions for DMDA1d with periodic boundary conditions.\n\n";

#include <petscdmda.h>

int main(int argc,char **argv)
{
  DM               da;
  Mat              A;
  const PetscInt   dfill[4] = {0,1,0,1},ofill[4] = {0,1,1,0};

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_PERIODIC,7,2,1,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMDASetBlockFills(da,dfill,ofill));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateMatrix(da,&A));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO));
  PetscCall(MatView(A,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(MatDestroy(&A));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2

TEST*/
