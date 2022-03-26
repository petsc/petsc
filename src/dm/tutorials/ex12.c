
static char help[] = "Tests DMGetGlobalVector() and DMRestoreGlobalVector().\n\n";

/*
Use the options
     -da_grid_x <nx> - number of grid points in x direction, if M < 0
     -da_grid_y <ny> - number of grid points in y direction, if N < 0
     -da_processors_x <MX> number of processors in x directio
     -da_processors_y <MY> number of processors in x direction
*/

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt         M = 10,N = 8;
  PetscBool        flg = PETSC_FALSE;
  DM               da;
  Vec              global1,global2,global3;
  DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-star_stencil",&flg,NULL));
  if (flg) stype = DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMGetGlobalVector(da,&global1));
  PetscCall(DMGetGlobalVector(da,&global2));
  PetscCall(DMRestoreGlobalVector(da,&global1));
  PetscCall(DMRestoreGlobalVector(da,&global2));
  PetscCall(DMGetGlobalVector(da,&global1));
  PetscCall(DMGetGlobalVector(da,&global3));
  PetscCall(DMGetGlobalVector(da,&global2));
  PetscCall(DMRestoreGlobalVector(da,&global1));
  PetscCall(DMRestoreGlobalVector(da,&global3));
  PetscCall(DMRestoreGlobalVector(da,&global2));
  PetscCall(DMGetGlobalVector(da,&global1));
  PetscCall(DMGetGlobalVector(da,&global3));
  PetscCall(DMGetGlobalVector(da,&global2));
  PetscCall(DMRestoreGlobalVector(da,&global1));
  PetscCall(DMRestoreGlobalVector(da,&global3));
  PetscCall(DMRestoreGlobalVector(da,&global2));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

TEST*/
