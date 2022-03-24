
static char help[] = "Tests mirror boundary conditions in 3-d.\n\n";

#include <petscdm.h>
#include <petscdmda.h>

int main(int argc,char **argv)
{
  PetscInt       M = 2, N = 3, P = 4,stencil_width = 1, dof = 1,m,n,p,xstart,ystart,zstart,i,j,k,c;
  DM             da;
  Vec            global,local;
  PetscScalar    ****vglobal;
  PetscViewer    sview;
  PetscScalar    sum;

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-stencil_width",&stencil_width,0));
  CHKERRQ(PetscOptionsGetInt(NULL,0,"-dof",&dof,0));

  CHKERRQ(DMDACreate3d(PETSC_COMM_WORLD,DM_BOUNDARY_MIRROR,DM_BOUNDARY_MIRROR,DM_BOUNDARY_MIRROR,DMDA_STENCIL_STAR,M,N,P,PETSC_DECIDE,PETSC_DECIDE,PETSC_DECIDE,dof,stencil_width,NULL,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDAGetCorners(da,&xstart,&ystart,&zstart,&m,&n,&p));

  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMDAVecGetArrayDOF(da,global,&vglobal));
  for (k=zstart; k<zstart+p; k++) {
    for (j=ystart; j<ystart+n; j++) {
      for (i=xstart; i<xstart+m; i++) {
        for (c=0; c<dof; c++) {
          vglobal[k][j][i][c] = 1000*k + 100*j + 10*i + c;
        }
      }
    }
  }
  CHKERRQ(DMDAVecRestoreArrayDOF(da,global,&vglobal));

  CHKERRQ(DMCreateLocalVector(da,&local));
  CHKERRQ(DMGlobalToLocalBegin(da,global,ADD_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,ADD_VALUES,local));

  CHKERRQ(VecSum(local,&sum));
  CHKERRQ(PetscSynchronizedPrintf(PETSC_COMM_WORLD,"sum %g\n",(double)PetscRealPart(sum)));
  CHKERRQ(PetscSynchronizedFlush(PETSC_COMM_WORLD,stdout));
  CHKERRQ(PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sview));
  CHKERRQ(VecView(local,sview));
  CHKERRQ(PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,&sview));
  CHKERRQ(PetscViewerFlush(PETSC_VIEWER_STDOUT_WORLD));

  CHKERRQ(DMDestroy(&da));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(VecDestroy(&global));

  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:

   test:
     suffix: 2
     nsize: 3

TEST*/
