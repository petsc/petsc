
static char help[] = "Tests saving DMDA vectors to files.\n\n";

/*
    ex13.c reads in the DMDA and vector written by this program.
*/

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscInt       M = 10,N = 8,m = PETSC_DECIDE,n = PETSC_DECIDE, dof = 1;
  PetscErrorCode ierr;
  DM             da;
  Vec            local,global,natural;
  PetscScalar    value;
  PetscViewer    bviewer;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,dof,1,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);

  value = -3.0;
  ierr  = VecSet(global,value);CHKERRQ(ierr);
  ierr  = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr  = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  ierr  = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  value = rank+1;
  ierr  = VecScale(local,value);CHKERRQ(ierr);
  ierr  = DMLocalToGlobalBegin(da,local,ADD_VALUES,global);CHKERRQ(ierr);
  ierr  = DMLocalToGlobalEnd(da,local,ADD_VALUES,global);CHKERRQ(ierr);

  ierr = DMDACreateNaturalVector(da,&natural);CHKERRQ(ierr);
  ierr = DMDAGlobalToNaturalBegin(da,global,INSERT_VALUES,natural);CHKERRQ(ierr);
  ierr = DMDAGlobalToNaturalEnd(da,global,INSERT_VALUES,natural);CHKERRQ(ierr);

  ierr = DMDASetFieldName(da,0,"First field");CHKERRQ(ierr);
  /*  ierr = VecView(global,PETSC_VIEWER_DRAW_WORLD);CHKERRQ(ierr); */

  ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,"daoutput",FILE_MODE_WRITE,&bviewer);CHKERRQ(ierr);
  ierr = DMView(da,bviewer);CHKERRQ(ierr);
  ierr = VecView(global,bviewer);CHKERRQ(ierr);
  ierr = PetscViewerDestroy(&bviewer);CHKERRQ(ierr);

  /* Free memory */
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = VecDestroy(&natural);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

