
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
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer));

  /* Read options */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,1,1,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));

  value = -3.0;
  CHKERRQ(VecSet(global,value));
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  value = rank+1;
  CHKERRQ(VecScale(local,value));
  CHKERRQ(DMLocalToGlobalBegin(da,local,ADD_VALUES,global));
  CHKERRQ(DMLocalToGlobalEnd(da,local,ADD_VALUES,global));

  CHKERRQ(VecView(global,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(DMView(da,viewer));

  /* Free memory */
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&da));
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
