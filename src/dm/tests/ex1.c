
static char help[] = "Tests various DM routines.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscInt       M = 10,N = 8,m = PETSC_DECIDE,n = PETSC_DECIDE;
  PetscErrorCode ierr;
  DM             da;
  PetscViewer    viewer;
  Vec            local,global;
  PetscScalar    value;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,1,1,NULL,NULL,&da);CHKERRQ(ierr);
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

  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = DMView(da,viewer);CHKERRQ(ierr);

  /* Free memory */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:
      nsize: 2
      args: -nox
      filter: grep -v -i Object

   test:
      suffix: cuda1
      requires: cuda
      args: -dm_vec_type cuda -nox
      filter: grep -v -i Object

   test:
      suffix: cuda2
      nsize: 2
      requires: cuda
      args: -dm_vec_type cuda -nox
      filter: grep -v -i Object

TEST*/
