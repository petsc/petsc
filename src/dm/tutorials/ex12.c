
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
  PetscErrorCode   ierr;
  PetscBool        flg = PETSC_FALSE;
  DM               da;
  Vec              global1,global2,global3;
  DMBoundaryType   bx    = DM_BOUNDARY_NONE,by = DM_BOUNDARY_NONE;
  DMDAStencilType  stype = DMDA_STENCIL_BOX;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetBool(NULL,NULL,"-star_stencil",&flg,NULL);CHKERRQ(ierr);
  if (flg) stype = DMDA_STENCIL_STAR;

  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,bx,by,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&global1);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&global2);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&global1);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&global2);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&global1);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&global3);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&global2);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&global1);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&global3);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&global2);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&global1);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&global3);CHKERRQ(ierr);
  ierr = DMGetGlobalVector(da,&global2);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&global1);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&global3);CHKERRQ(ierr);
  ierr = DMRestoreGlobalVector(da,&global2);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

TEST*/
