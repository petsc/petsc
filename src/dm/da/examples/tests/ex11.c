/*$Id: ex11.c,v 1.10 2000/01/11 21:03:26 bsmith Exp bsmith $*/

static char help[] = "Tests various 1-dimensional DA routines.\n\n";

#include "da.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int    M = 5,N = 4,ierr,dof=1,s=1,wrap=0,i,n,j,k,m,cnt;
  DA     da;
  Viewer viewer;
  Vec    local,locala,global,coors;
  Scalar *xy,*alocal;
  Draw   draw;
  char   fname[16];

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Create viewers */
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",PETSC_DECIDE,PETSC_DECIDE,600,200,&viewer);CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw);CHKERRA(ierr);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-N",&N,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-dof",&dof,PETSC_NULL);CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-periodic",&wrap,PETSC_NULL);CHKERRA(ierr); 

  /* Create distributed array and get vectors */
  ierr = DACreate2d(PETSC_COMM_WORLD,(DAPeriodicType)wrap,DA_STENCIL_BOX,M,N,PETSC_DECIDE,
                    PETSC_DECIDE,dof,s,PETSC_NULL,PETSC_NULL,&da);CHKERRA(ierr);
  ierr = DASetUniformCoordinates(da,0.0,1.0,0.0,1.0,0.0,0.0);CHKERRA(ierr);
  for (i=0; i<dof; i++) {
    sprintf(fname,"Field %d",i);
    ierr = DASetFieldName(da,i,fname);CHKERRA(ierr);
  }

  ierr = DAView(da,viewer);CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&locala);CHKERRA(ierr);
  ierr = DAGetCoordinates(da,&coors);CHKERRA(ierr);
  ierr = VecGetArray(coors,&xy);CHKERRA(ierr);

ierr = VecView(coors,VIEWER_STDOUT_SELF);

  /* Set values into local vectors */
  ierr = VecGetArray(local,&alocal);CHKERRA(ierr);
  ierr = DAGetGhostCorners(da,0,0,0,&m,&n,0);CHKERRA(ierr);
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
  ierr = VecRestoreArray(local,&alocal);CHKERRA(ierr);
  ierr = VecRestoreArray(coors,&xy);CHKERRA(ierr);

  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global);CHKERRA(ierr);

  ierr = VecView(global,viewer);CHKERRA(ierr); 

  /* Send ghost points to local vectors */
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,locala);CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,locala);CHKERRA(ierr);

  /* Free memory */
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  ierr = VecDestroy(global);CHKERRA(ierr);
  ierr = VecDestroy(local);CHKERRA(ierr);
  ierr = VecDestroy(locala);CHKERRA(ierr);
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 









