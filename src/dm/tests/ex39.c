
static char help[] = "Tests mirror boundary conditions in 1-d.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt       M = 6,stencil_width = 1, dof = 1,m,xstart,i,j;
  DM             da;
  Vec            global,local;
  PetscScalar    **vglobal;
  PetscViewer    sviewer;

  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  PetscCall(PetscOptionsGetInt(NULL,0,"-stencil_width",&stencil_width,0));
  PetscCall(PetscOptionsGetInt(NULL,0,"-dof",&dof,0));

  PetscCall(DMDACreate1d(PETSC_COMM_WORLD,DM_BOUNDARY_MIRROR,M,dof,stencil_width,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDAGetCorners(da,&xstart,0,0,&m,0,0));

  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(DMDAVecGetArrayDOF(da,global,&vglobal));
  for (i=xstart; i<xstart+m; i++) {
    for (j=0; j<dof; j++) {
      vglobal[i][j] = 100*(i+1) + j;
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(da,global,&vglobal));

  PetscCall(DMCreateLocalVector(da,&local));
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  PetscCall(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(VecView(local,sviewer));
  PetscCall(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sviewer));
  PetscCall(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(global,PETSC_VIEWER_STDOUT_WORLD));

  PetscCall(DMDestroy(&da));
  PetscCall(VecDestroy(&local));
  PetscCall(VecDestroy(&global));

  PetscCall(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
      suffix: 2
      nsize: 2
      filter: grep -v "Vec Object"

TEST*/
