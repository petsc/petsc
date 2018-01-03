
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
  ierr = PetscOptionsGetInt(NULL,0,"-stencil_width",&stencil_width,0);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,0,"-dof",&dof,0);CHKERRQ(ierr);

  ierr = DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_MIRROR,M,dof,stencil_width,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xstart,0,0,&m,0,0);CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,global,&vglobal);CHKERRQ(ierr);
  for (i=xstart; i<xstart+m; i++) {
    for (j=0; j<dof; j++) {
      vglobal[i][j] = 100*(i+1) + j;
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da,global,&vglobal);CHKERRQ(ierr);

  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  ierr = PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = VecView(local,sviewer);CHKERRQ(ierr);
  ierr = PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer);CHKERRQ(ierr);
  ierr = PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);

  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);

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
