
static char help[] = "Tests various 2-dimensional DMDA routines.\n\n";

#include <petscdmda.h>
#include <petscdraw.h>

int main(int argc,char **argv)
{
  PetscInt       M = 10,N = 8,dof=1,s=1,bx=0,by=0,i,n,j,k,m,wrap,xs,ys;
  PetscErrorCode ierr;
  DM             da,dac;
  PetscViewer    viewer;
  Vec            local,global,coors;
  PetscScalar    ***xy,***aglobal;
  PetscDraw      draw;
  char           fname[32];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);if (ierr) return ierr;
  /* Create viewers */
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",PETSC_DECIDE,PETSC_DECIDE,600,200,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(NULL,NULL,"-M",&M,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-N",&N,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-dof",&dof,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-s",&s,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-periodic_x",&wrap,NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL,NULL,"-periodic_y",&wrap,NULL);CHKERRQ(ierr);

  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,(DMBoundaryType)bx,(DMBoundaryType)by,DMDA_STENCIL_BOX,M,N,PETSC_DECIDE,PETSC_DECIDE,dof,s,NULL,NULL,&da);CHKERRQ(ierr);
  ierr = DMSetFromOptions(da);CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  for (i=0; i<dof; i++) {
    sprintf(fname,"Field %d",(int)i);
    ierr = DMDASetFieldName(da,i,fname);CHKERRQ(ierr);
  }

  ierr = DMView(da,viewer);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);
  ierr = DMGetCoordinates(da,&coors);CHKERRQ(ierr);
  ierr = DMGetCoordinateDM(da,&dac);CHKERRQ(ierr);

  /* Set values into global vectors */
  ierr = DMDAVecGetArrayDOFRead(dac,coors,&xy);CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da,global,&aglobal);CHKERRQ(ierr);
  ierr = DMDAGetCorners(da,&xs,&ys,0,&m,&n,0);CHKERRQ(ierr);
  for (k=0; k<dof; k++) {
    for (j=ys; j<ys+n; j++) {
      for (i=xs; i<xs+m; i++) {
        aglobal[j][i][k] = PetscSinScalar(2.0*PETSC_PI*(k+1)*xy[j][i][0]);
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da,global,&aglobal);CHKERRQ(ierr);
  ierr = DMDAVecRestoreArrayDOFRead(dac,coors,&xy);CHKERRQ(ierr);
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRQ(ierr);

  ierr = VecSet(global,0.0);CHKERRQ(ierr);
  ierr = DMLocalToGlobalBegin(da,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRQ(ierr);
  ierr = VecView(global,viewer);CHKERRQ(ierr);

  /* Free memory */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return ierr;
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
