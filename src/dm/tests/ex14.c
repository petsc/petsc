
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
  DM             da;
  Vec            local,global,natural;
  PetscScalar    value;
  PetscViewer    bviewer;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-m",&m,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-n",&n,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,DMDA_STENCIL_BOX,M,N,m,n,dof,1,NULL,NULL,&da));
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

  CHKERRQ(DMDACreateNaturalVector(da,&natural));
  CHKERRQ(DMDAGlobalToNaturalBegin(da,global,INSERT_VALUES,natural));
  CHKERRQ(DMDAGlobalToNaturalEnd(da,global,INSERT_VALUES,natural));

  CHKERRQ(DMDASetFieldName(da,0,"First field"));
  /*  CHKERRQ(VecView(global,PETSC_VIEWER_DRAW_WORLD)); */

  CHKERRQ(PetscViewerBinaryOpen(PETSC_COMM_WORLD,"daoutput",FILE_MODE_WRITE,&bviewer));
  CHKERRQ(DMView(da,bviewer));
  CHKERRQ(VecView(global,bviewer));
  CHKERRQ(PetscViewerDestroy(&bviewer));

  /* Free memory */
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(VecDestroy(&natural));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}
