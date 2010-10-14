
static char help[] = "Tests DMGetGlobalVector() and DMRestoreGlobalVector().\n\n";

/*
Use the options
     -da_grid_x <nx> - number of grid points in x direction, if M < 0
     -da_grid_y <ny> - number of grid points in y direction, if N < 0
     -da_processors_x <MX> number of processors in x directio
     -da_processors_y <MY> number of processors in x direction
*/

#include "petscda.h"

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       M = -10,N = -8;
  PetscErrorCode ierr;
  PetscBool      flg = PETSC_FALSE;
  DM             da;
  Vec            global1,global2,global3;
  DAPeriodicType ptype = DA_NONPERIODIC;
  DAStencilType  stype = DA_STENCIL_BOX;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 
  ierr = PetscOptionsGetTruth(PETSC_NULL,"-star_stencil",&flg,PETSC_NULL);CHKERRQ(ierr);
  if (flg) stype = DA_STENCIL_STAR;
      
  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,ptype,stype,M,N,PETSC_DECIDE,PETSC_DECIDE,1,1,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
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
  ierr = DMDestroy(da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 
