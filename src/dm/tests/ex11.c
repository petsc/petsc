
static char help[] = "Tests various 2-dimensional DMDA routines.\n\n";

#include <petscdmda.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscInt       M = 10,N = 8,dof=1,s=1,bx=0,by=0,i,n,j,k,m,wrap,xs,ys;
  DM             da,dac;
  PetscViewer    viewer;
  Vec            local,global,coors;
  PetscScalar    ***xy,***aglobal;
  PetscDraw      draw;
  char           fname[32];

  CHKERRQ(PetscInitialize(&argc,&argv,(char*)0,help));
  /* Create viewers */
  CHKERRQ(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",PETSC_DECIDE,PETSC_DECIDE,600,200,&viewer));
  CHKERRQ(PetscViewerDrawGetDraw(viewer,0,&draw));
  CHKERRQ(PetscDrawSetDoubleBuffer(draw));

  /* Read options */
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-periodic_x",&wrap,NULL));
  CHKERRQ(PetscOptionsGetInt(NULL,NULL,"-periodic_y",&wrap,NULL));

  /* Create distributed array and get vectors */
  CHKERRQ(DMDACreate2d(PETSC_COMM_WORLD,(DMBoundaryType)bx,(DMBoundaryType)by,DMDA_STENCIL_BOX,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,&da));
  CHKERRQ(DMSetFromOptions(da));
  CHKERRQ(DMSetUp(da));
  CHKERRQ(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,0.0));
  for (i=0; i<dof; i++) {
    sprintf(fname,"Field %d",(int)i);
    CHKERRQ(DMDASetFieldName(da,i,fname));
  }

  CHKERRQ(DMView(da,viewer));
  CHKERRQ(DMCreateGlobalVector(da,&global));
  CHKERRQ(DMCreateLocalVector(da,&local));
  CHKERRQ(DMGetCoordinates(da,&coors));
  CHKERRQ(DMGetCoordinateDM(da,&dac));

  /* Set values into global vectors */
  CHKERRQ(DMDAVecGetArrayDOFRead(dac,coors,&xy));
  CHKERRQ(DMDAVecGetArrayDOF(da,global,&aglobal));
  CHKERRQ(DMDAGetCorners(da,&xs,&ys,0,&m,&n,0));
  for (k=0; k<dof; k++) {
    for (j=ys; j<ys+n; j++) {
      for (i=xs; i<xs+m; i++) {
        aglobal[j][i][k] = PetscSinScalar(2.0*PETSC_PI*(k+1)*xy[j][i][0]);
      }
    }
  }
  CHKERRQ(DMDAVecRestoreArrayDOF(da,global,&aglobal));
  CHKERRQ(DMDAVecRestoreArrayDOFRead(dac,coors,&xy));
  CHKERRQ(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  CHKERRQ(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  CHKERRQ(VecSet(global,0.0));
  CHKERRQ(DMLocalToGlobalBegin(da,local,INSERT_VALUES,global));
  CHKERRQ(DMLocalToGlobalEnd(da,local,INSERT_VALUES,global));
  CHKERRQ(VecView(global,PETSC_VIEWER_STDOUT_WORLD));
  CHKERRQ(VecView(global,viewer));

  /* Free memory */
  CHKERRQ(PetscViewerDestroy(&viewer));
  CHKERRQ(VecDestroy(&global));
  CHKERRQ(VecDestroy(&local));
  CHKERRQ(DMDestroy(&da));
  CHKERRQ(PetscFinalize());
  return 0;
}

/*TEST

   test:
      args: -dof 2
      filter: grep -v -i Object
      requires: x

   test:
      suffix: 2
      nsize: 2
      args: -dof 2
      filter: grep -v -i Object
      requires: x

   test:
      suffix: 3
      nsize: 3
      args: -dof 2
      filter: grep -v -i Object
      requires: x

TEST*/
