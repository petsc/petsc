
static char help[] = "Tests mirror boundary conditions in 1-d.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscErrorCode ierr;
  PetscInt       M = 6,stencil_width = 1, dof = 1,m,xstart,i,j;
  DM             da;
  Vec            global,local;
  PetscScalar    **vglobal;
  PetscViewer    sviewer;

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-stencil_width",&stencil_width,0));
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-dof",&dof,0));

  CHKERRQ(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_MIRROR,M,dof,stencil_width,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDAGetCorners(da,&xstart,0,0,&m,0,0));

  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMDAVecGetArrayDOF(da,global,&vglobal));
  for (i=xstart; i<xstart+m; i++) {
    for (j=0; j<dof; j++) {
      vglobal[i][j] = 100*(i+1) + j;
    }
  }
  CHKERRQ(DMDAVecRestoreArrayDOF(da,global,&vglobal));

  CHKERRQ(DMCreateLocalVector(da,&local));
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(VecView(local,sviewer));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(global,PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(DMDestroy(&da));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));

  ierr = PetscFinalize();
  return ierr;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2
      filter: grep -v "Vec Object"

TEST*/
