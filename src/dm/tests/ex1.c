
static char help[] = "Tests various DM routines.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscMPIInt    rank;
  PetscInt       M = 10,N = 8,m = PETSC_DECIDE,n = PETSC_DECIDE;
  DM             da;
  PetscViewer    viewer;
  Vec            local,global;
  PetscScalar    value;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",300,0,300,300,&viewer));

  /* Read options */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,1,1,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(DMCreateLocalVector(da,&local));

  value = -3.0;
  PetscCall(VecSet(global,value));
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  value = rank+1;
  PetscCall(VecScale(local,value));
  PetscCall(DMLocalToGlobalBegin(da,local,ADD_VALUES,global));
  PetscCall(DMLocalToGlobalEnd(da,local,ADD_VALUES,global));

  PetscCall(VecView(global,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(DMView(da,viewer));

  /* Free memory */
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
  return 0;
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
