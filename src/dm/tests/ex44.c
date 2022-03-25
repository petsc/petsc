
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

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));

  PetscCall(DMCompositeCreate(PETSC_COMM_WORLD,&packer));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,1,1,NULL,&da1));
  PetscCall(DMSetFromOptions(da1));
  PetscCall(DMSetUp(da1));
  PetscCall(DMCompositeAddDM(packer,da1));
  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,6,1,1,NULL,&da2));
  PetscCall(DMSetFromOptions(da2));
  PetscCall(DMSetUp(da2));
  PetscCall(DMCompositeAddDM(packer,da2));

  PetscCall(DMCreateGlobalVector(packer,&global));
  PetscCall(DMCreateLocalVector(packer,&local));
  PetscCall(DMCreateLocalVector(packer,&buffer));

  PetscCall(DMCompositeGetAccessArray(packer,global,2,NULL,globals));
  value = 1;
  PetscCall(VecSet(globals[0], value));
  value = -1;
  PetscCall(VecSet(globals[1], value));
  PetscCallMPI(MPI_Comm_rank(PETSC_COMM_WORLD,&rank));
  value = rank + 1;
  PetscCall(VecScale(globals[0], value));
  PetscCall(VecScale(globals[1], value));
  PetscCall(DMCompositeRestoreAccessArray(packer,global,2,NULL,globals));

  /* Test GlobalToLocal in insert mode */
  PetscCall(DMGlobalToLocalBegin(packer,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(packer,global,INSERT_VALUES,local));

  PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal Vector: processor %d\n",rank));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer));
  PetscCall(VecView(local,viewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  /* Test LocalToGlobal in insert mode */
  PetscCall(DMLocalToGlobalBegin(packer,local,INSERT_VALUES,global));
  PetscCall(DMLocalToGlobalEnd(packer,local,INSERT_VALUES,global));

  PetscCall(VecView(global,PETSC_VIEWER_STDOUT_WORLD));

  /* Test LocalToLocal in insert mode */
  PetscCall(DMLocalToLocalBegin(packer,local,INSERT_VALUES,buffer));
  PetscCall(DMLocalToLocalEnd(packer,local,INSERT_VALUES,buffer));

  PetscCall(PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal Vector: processor %d\n",rank));
  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer));
  PetscCall(VecView(buffer,viewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(VecDestroy(&buffer));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));
  PetscCall(DMDestroy(&packer));
  PetscCall(DMDestroy(&da2));
  PetscCall(DMDestroy(&da1));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:
      nsize: 3

TEST*/
