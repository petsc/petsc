static char help[] = "Test DMClone_Stag()\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  PetscErrorCode  ierr;
  DM              dm,dm2;
  PetscInt        dim;
  PetscBool       flg,setSizes;

  /* Create a DMStag object */
  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-dim",&dim,&flg);CHKERRQ(ierr);
  if (!flg) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Supply -dim option with value 1, 2, or 3");
  setSizes = PETSC_FALSE;
  ierr = PetscOptionsGetBool(NULL,NULL,"-setsizes",&setSizes,NULL);CHKERRQ(ierr);
  if (setSizes) {
    PetscMPIInt size;
    PetscInt lx[4] = {2,3},   ranksx = 2, mx = 5;
    PetscInt ly[3] = {3,8,2}, ranksy = 3, my = 13;
    PetscInt lz[2] = {2,4},   ranksz = 2, mz = 6;

    ierr = MPI_Comm_size(PETSC_COMM_WORLD,&size);CHKERRMPI(ierr);
    switch (dim) {
      case 1:
        if (size != ranksx) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %D ranks with -dim 1 -setSizes",ranksx);
        ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,mx,1,1,DMSTAG_STENCIL_BOX,1,lx,&dm);CHKERRQ(ierr);
        break;
      case 2:
        if (size != ranksx * ranksy) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %D ranks with -dim 2 -setSizes",ranksx * ranksy);
        ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,mx,my,ranksx,ranksy,1,1,1,DMSTAG_STENCIL_BOX,1,lx,ly,&dm);CHKERRQ(ierr);
        break;
      case 3:
        if (size != ranksx * ranksy * ranksz) SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %D ranks with -dim 3 -setSizes", ranksx * ranksy * ranksz);
        ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,mx,my,mz,ranksx,ranksy,ranksz,1,1,1,1,DMSTAG_STENCIL_BOX,1,lx,ly,lz,&dm);CHKERRQ(ierr);
        break;
      default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %D",dim);
    }
  } else {
    if (dim == 1) {
      ierr = DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,2,2,3,DMSTAG_STENCIL_BOX,1,NULL,&dm);CHKERRQ(ierr);
    } else if (dim == 2) {
      ierr = DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,2,2,PETSC_DECIDE,PETSC_DECIDE,2,3,4,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm);CHKERRQ(ierr);
    } else if (dim == 3) {
      ierr = DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,2,2,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,5,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm);CHKERRQ(ierr);
    } else {
      ierr = PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option with value 1, 2, or 3\n");CHKERRQ(ierr);
      return 1;
    }
  }
  ierr = DMSetFromOptions(dm);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = DMSetUp(dm);CHKERRQ(ierr);CHKERRQ(ierr);
  ierr = DMView(dm,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Create a cloned DMStag object */
  ierr = DMClone(dm,&dm2);CHKERRQ(ierr);
  ierr = DMView(dm2,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMDestroy(&dm);CHKERRQ(ierr);
  ierr = DMDestroy(&dm2);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      suffix: 1
      nsize: 1
      args: -dim 1

   test:
      suffix: 2
      nsize: 4
      args: -dim 2

   test:
      suffix: 3
      nsize: 6
      args: -dim 3 -stag_grid_x 3 -stag_grid_y 2 -stag_grid_z 1

   test:
      suffix: 4
      nsize: 2
      args: -dim 1 -setsizes

   test:
      suffix: 5
      nsize: 6
      args: -dim 2 -setsizes

   test:
      suffix: 6
      nsize: 12
      args: -dim 3 -setsizes

TEST*/
