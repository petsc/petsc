
static char help[] = "Tests various 2-dimensional DM routines.\n\n";

#include <petscdmda.h>

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **argv)
{
  PetscInt       M = 5,N = 4,dof=1,s=1,bx=0,by=0,i,n,j,k,m,cnt;
  PetscErrorCode ierr;
  DM             da;
  PetscViewer    viewer;
  Vec            local,locala,global,coors;
  PetscScalar    *xy,*alocal;
  PetscDraw      draw;
  char           fname[16];

  ierr = PetscInitialize(&argc,&argv,(char*)0,help);CHKERRQ(ierr); 

  /* Create viewers */
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",PETSC_DECIDE,PETSC_DECIDE,600,200,&viewer);CHKERRQ(ierr);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRQ(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRQ(ierr);

  /* Read options */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-periodic_x",&wrap,PETSC_NULL);CHKERRQ(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-periodic_y",&wrap,PETSC_NULL);CHKERRQ(ierr); 

  /* Create distributed array and get vectors */
  ierr = DMDACreate2d(PETSC_COMM_WORLD,(DMDABoundaryType)bx,(DMDABoundaryType)by,DMDA_STENCIL_BOX,M,N,PETSC_DECIDE,
                    PETSC_DECIDE,dof,s,PETSC_NULL,PETSC_NULL,&da);CHKERRQ(ierr);
  ierr = DMDASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRQ(ierr);
  for (i=0; i<dof; i++) {
    sprintf(fname,"Field %d",(int)i);
    ierr = DMDASetFieldName(da,i,fname);CHKERRQ(ierr);
  }

  ierr = DMView(da,viewer);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da,&global);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&local);CHKERRQ(ierr);
  ierr = DMCreateLocalVector(da,&locala);CHKERRQ(ierr);
  ierr = DMDAGetCoordinates(da,&coors);CHKERRQ(ierr);
  ierr = VecGetArray(coors,&xy);CHKERRQ(ierr);

  ierr = VecView(coors,PETSC_VIEWER_STDOUT_SELF);

  /* Set values into local vectors */
  ierr = VecGetArray(local,&alocal);CHKERRQ(ierr);
  ierr = DMDAGetGhostCorners(da,0,0,0,&m,&n,0);CHKERRQ(ierr);
  n    = n/dof;
  for (k=0; k<dof; k++) {
    cnt = 0;
    for (j=0; j<n; j++) {
      for (i=0; i<m; i++) {
        alocal[k+dof*cnt] = PetscSinScalar(2.0*PETSC_PI*(k+1)*xy[2*cnt]);
        cnt++;
      }
    }
  }
  ierr = VecRestoreArray(local,&alocal);CHKERRQ(ierr);
  ierr = VecRestoreArray(coors,&xy);CHKERRQ(ierr);

  ierr = DMLocalToGlobalBegin(da,local,INSERT_VALUES,global);CHKERRQ(ierr);
  ierr = DMLocalToGlobalEnd(da,local,INSERT_VALUES,global);CHKERRQ(ierr);

  ierr = VecView(global,viewer);CHKERRQ(ierr); 

  /* Send ghost points to local vectors */
  ierr = DMGlobalToLocalBegin(da,global,INSERT_VALUES,locala);CHKERRQ(ierr);
  ierr = DMGlobalToLocalEnd(da,global,INSERT_VALUES,locala);CHKERRQ(ierr);

  /* Free memory */
  ierr = PetscViewerDestroy(&viewer);CHKERRQ(ierr);
  ierr = VecDestroy(&global);CHKERRQ(ierr);
  ierr = VecDestroy(&local);CHKERRQ(ierr);
  ierr = VecDestroy(&locala);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
  ierr = PetscFinalize();
  return 0;
}
 









