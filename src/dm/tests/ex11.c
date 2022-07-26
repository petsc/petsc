
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

  PetscFunctionBeginUser;
  PetscCall(PetscInitialize(&argc,&argv,(char*)0,help));
  /* Create viewers */
  PetscCall(PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",PETSC_DECIDE,PETSC_DECIDE,600,200,&viewer));
  PetscCall(PetscViewerDrawGetDraw(viewer,0,&draw));
  PetscCall(PetscDrawSetDoubleBuffer(draw));

  /* Read options */
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-periodic_x",&wrap,NULL));
  PetscCall(PetscOptionsGetInt(NULL,NULL,"-periodic_y",&wrap,NULL));

  /* Create distributed array and get vectors */
  PetscCall(DMDACreate2d(PETSC_COMM_WORLD,(DMBoundaryType)bx,(DMBoundaryType)by,DMDA_STENCIL_BOX,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,&da));
  PetscCall(DMSetFromOptions(da));
  PetscCall(DMSetUp(da));
  PetscCall(DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,0.0));
  for (i=0; i<dof; i++) {
    sprintf(fname,"Field %d",(int)i);
    PetscCall(DMDASetFieldName(da,i,fname));
  }

  PetscCall(DMView(da,viewer));
  PetscCall(DMCreateGlobalVector(da,&global));
  PetscCall(DMCreateLocalVector(da,&local));
  PetscCall(DMGetCoordinates(da,&coors));
  PetscCall(DMGetCoordinateDM(da,&dac));

  /* Set values into global vectors */
  PetscCall(DMDAVecGetArrayDOFRead(dac,coors,&xy));
  PetscCall(DMDAVecGetArrayDOF(da,global,&aglobal));
  PetscCall(DMDAGetCorners(da,&xs,&ys,0,&m,&n,0));
  for (k=0; k<dof; k++) {
    for (j=ys; j<ys+n; j++) {
      for (i=xs; i<xs+m; i++) {
        aglobal[j][i][k] = PetscSinScalar(2.0*PETSC_PI*(k+1)*xy[j][i][0]);
      }
    }
  }
  PetscCall(DMDAVecRestoreArrayDOF(da,global,&aglobal));
  PetscCall(DMDAVecRestoreArrayDOFRead(dac,coors,&xy));
  PetscCall(DMGlobalToLocalBegin(da,global,INSERT_VALUES,local));
  PetscCall(DMGlobalToLocalEnd(da,global,INSERT_VALUES,local));

  PetscCall(VecSet(global,0.0));
  PetscCall(DMLocalToGlobalBegin(da,local,INSERT_VALUES,global));
  PetscCall(DMLocalToGlobalEnd(da,local,INSERT_VALUES,global));
  PetscCall(VecView(global,PETSC_VIEWER_STDOUT_WORLD));
  PetscCall(VecView(global,viewer));

  /* Free memory */
  PetscCall(PetscViewerDestroy(&viewer));
  PetscCall(VecDestroy(&global));
  PetscCall(VecDestroy(&local));
  PetscCall(DMDestroy(&da));
  PetscCall(PetscFinalize());
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
