
static char help[] = "Tests various DMComposite routines.\n\n";

#include <petscdm.h>
#include <petscdmda.h>
#include <petscdmcomposite.h>

int main(int argc,char **argv)
{
  PetscMPIInt            rank;
  PetscErrorCode         ierr;
  DM                     da1,da2,packer;
  Vec                    local,global,globals[2],buffer;
  PetscScalar            value;
  PetscViewer            viewer;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;

  ierr = DMCompositeCreate(PETSC_COMM_WORLD,&packer);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,8,1,1,NULL,&da1);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da1);CHKERRQ(ierr);
  ierr = DMSetUp(da1);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,da1);CHKERRQ(ierr);
  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_NONE,6,1,1,NULL,&da2);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da2);CHKERRQ(ierr);
  ierr = DMSetUp(da2);CHKERRQ(ierr);
  ierr = DMCompositeAddDM(packer,da2);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(packer,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(packer,&local);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(packer,&buffer);CHKERRQ(ierr);

  ierr = DMCompositeGetAccessArray(packer,global,2,NULL,globals);CHKERRQ(ierr);
  value = 1;
  ierr = VecSet(globals[0], value);CHKERRQ(ierr);
  value = -1;
  ierr = VecSet(globals[1], value);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRMPI(ierr);
  value = rank + 1;
  ierr = VecScale(globals[0], value);CHKERRQ(ierr);
  ierr = VecScale(globals[1], value);CHKERRQ(ierr);
  ierr = DMCompositeRestoreAccessArray(packer,global,2,NULL,globals);CHKERRQ(ierr);

  /* Test GlobalToLocal in insert mode */
  ierr = DMGlobalToLocalBegin(packer,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(packer,global,INSERT_VALUES,local);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal Vector: processor %d\n",rank);CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer);CHKERRQ(ierr);
  ierr = VecView(local,viewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test LocalToGlobal in insert mode */
  ierr = DMLocalToGlobalBegin(packer,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(packer,local,INSERT_VALUES,global);CHKERRQ(ierr);

  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  /* Test LocalToLocal in insert mode */
  ierr = DMLocalToLocalBegin(packer,local,INSERT_VALUES,buffer);CHKERRQ(ierr);
  ierr = DMLocalToLocalEnd(packer,local,INSERT_VALUES,buffer);CHKERRQ(ierr);

  ierr = PetscViewerASCIIPushSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIISynchronizedPrintf(PETSC_VIEWER_STDOUT_WORLD,"\nLocal Vector: processor %d\n",rank);CHKERRQ(ierr);
  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer);CHKERRQ(ierr);
  ierr = VecView(buffer,viewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopSynchronized(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = VecDestroy(&buffer);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = DMDestroy(&packer);CHKERRQ(ierr);
  ierr = DMDestroy(&da2);CHKERRQ(ierr);
  ierr = DMDestroy(&da1);CHKERRQ(ierr);

  ierr = PetscFinalize();

  return ierr;
}


/*TEST

   test:
      nsize: 3

TEST*/
