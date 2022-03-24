
static char help[] = "Tests various DMComposite routines.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>

int main(int argc,char **argv)
{
  PetscMPIInt rank;
  DM          da1,da2,packer;
  Vec         local,global,globals[2],buffer;
  PetscScalar value;
  PetscViewer viewer;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));

  CHKERRQ(DMCompositeCreate(PETSC_COMM_WORLD,&packer));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,1,1,NULL,&da1));
  CHKERRQ(DMSetFromOptions(da1));
  CHKERRQ(DMSetUp(da1));
  CHKERRQ(DMCompositeAddDM(packer,da1));
  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,6,1,1,NULL,&da2));
  CHKERRQ(DMSetFromOptions(da2));
  CHKERRQ(DMSetUp(da2));
  CHKERRQ(DMCompositeAddDM(packer,da2));

  CHKERRQ(DMCreateGlobalVector(packer,&global));
  CHKERRQ(DMCreateLocalVector(packer,&local));
  CHKERRQ(DMCreateLocalVector(packer,&buffer));

  CHKERRQ(DMCompositeGetAccessArray(packer,global,2,NULL,globals));
  value = 1;
  CHKERRQ(VecSet(globals[0], value));
  value = -1;
  CHKERRQ(VecSet(globals[1], value));
  CHKERRMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  value = rank + 1;
  CHKERRQ(VecScale(globals[0], value));
  CHKERRQ(VecScale(globals[1], value));
  CHKERRQ(DMCompositeRestoreAccessArray(packer,global,2,NULL,globals));

  /* Test GlobalToLocal in insert mode */
  CHKERRQ(DMGlobalToLocalBegin(packer,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(packer,global,INSERT_VALUES,local));

  CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal Vector: processor %d\n",rank));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer));
  CHKERRQ(VecView(local,viewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  /* Test LocalToGlobal in insert mode */
  CHKERRQ(DMLocalToGlobalBegin(packer,local,INSERT_VALUES,global));
  CHKERRQ(DMLocalToGlobalEnd(packer,local,INSERT_VALUES,global));

  CHKERRQ(VecView(global,PETSC_VIEWER_STDOUT_WORLD));

  /* Test LocalToLocal in insert mode */
  CHKERRQ(DMLocalToLocalBegin(packer,local,INSERT_VALUES,buffer));
  CHKERRQ(DMLocalToLocalEnd(packer,local,INSERT_VALUES,buffer));

  CHKERRQ(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal Vector: processor %d\n",rank));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer));
  CHKERRQ(VecView(buffer,viewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(VecDestroy(&buffer));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(DMDestroy(&packer));
  CHKERRQ(DMDestroy(&da2));
  CHKERRQ(DMDestroy(&da1));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3

TEST*/
