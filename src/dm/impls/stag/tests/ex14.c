static char help[] = "Test DMClone_Stag()\n\n";

#include <petscdm.h>
#include <petscdmstag.h>

int main(int argc,char **argv)
{
  DM              dm,dm2;
  PetscInt        dim;
  PetscBool       flg,setSizes;

  /* Create a DMStag object */
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dim",&dim,&flg));
  PetscCheck(flg,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Supply -dim option with value 1, 2, or 3");
  setSizes = PETSC_FALSE;
  PetscCall(PetscOptionsGetBool(NULL,NULL,"-setsizes",&setSizes,NULL));
  if (setSizes) {
    PetscMPIInt size;
    PetscInt lx[4] = {2,3},   ranksx = 2, mx = 5;
    PetscInt ly[3] = {3,8,2}, ranksy = 3, my = 13;
    PetscInt lz[2] = {2,4},   ranksz = 2, mz = 6;

    PetscCallMPI(MPI_Comm_size(PETSC_COMM_WORLD,&size));
    switch (dim) {
      case 1:
        PetscCheck(size == ranksx,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %" PetscInt_FMT " ranks with -dim 1 -setSizes",ranksx);
        PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,mx,1,1,DMSTAG_STENCIL_BOX,1,lx,&dm));
        break;
      case 2:
        PetscCheck(size == ranksx*ranksy,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %" PetscInt_FMT " ranks with -dim 2 -setSizes",ranksx * ranksy);
        PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,mx,my,ranksx,ranksy,1,1,1,DMSTAG_STENCIL_BOX,1,lx,ly,&dm));
        break;
      case 3:
        PetscCheck(size == ranksx*ranksy*ranksz,PETSC_COMM_WORLD,PETSC_ERR_ARG_WRONG,"Must run on %" PetscInt_FMT " ranks with -dim 3 -setSizes", ranksx * ranksy * ranksz);
        PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,mx,my,mz,ranksx,ranksy,ranksz,1,1,1,1,DMSTAG_STENCIL_BOX,1,lx,ly,lz,&dm));
        break;
      default:
        SETERRQ(PETSC_COMM_WORLD,PETSC_ERR_SUP,"No support for dimension %" PetscInt_FMT,dim);
    }
  } else {
    if (dim == 1) {
      PetscCall(DMStagCreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,2,2,3,DMSTAG_STENCIL_BOX,1,NULL,&dm));
    } else if (dim == 2) {
      PetscCall(DMStagCreate2d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,2,2,PETSC_DECIDE,PETSC_DECIDE,2,3,4,DMSTAG_STENCIL_BOX,1,NULL,NULL,&dm));
    } else if (dim == 3) {
      PetscCall(DMStagCreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,DM_BOUNDARY_NONE,2,2,2,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,2,3,4,5,DMSTAG_STENCIL_BOX,1,NULL,NULL,NULL,&dm));
    } else {
      PetscCall(PetscPrintf(PETSC_COMM_WORLD,"Supply -dim option with value 1, 2, or 3\n"));
      return 1;
    }
  }
  PetscCall(DMSetFromOptions(dm));
  PetscCall(DMSetUp(dm));
  PetscCall(DMView(dm,PETSC_VIEWER_STDOUT_WORLD));

  /* Create a cloned DMStag object */
  PetscCall(DMClone(dm,&dm2));
  PetscCall(DMView(dm2,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMDestroy(&dm));
  PetscCall(DMDestroy(&dm2));
  PetscCall(PetscFinalize());
  return 0;
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
