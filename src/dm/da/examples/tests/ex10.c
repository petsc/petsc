#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: ex10.c,v 1.6 1999/05/04 20:37:40 balay Exp balay $";
#endif

static char help[] = "Tests various 1-dimensional DA routines.\n\n";

#include "da.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int    M = 13, ierr, dof=1, s=1, wrap=0, flg, i, n, j;
  DA     da;
  Viewer viewer;
  Vec    local, locala, global,coors;
  Scalar *x,*alocal;
  Draw   draw;
  char   fname[16];
  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Create viewers */
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",PETSC_DECIDE,PETSC_DECIDE,600,200,&viewer);CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw);CHKERRA(ierr);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-dof",&dof,&flg); CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,&flg); CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-periodic",&wrap,&flg); CHKERRA(ierr); 

  /* Create distributed array and get vectors */
  ierr = DACreate1d(PETSC_COMM_WORLD,(DAPeriodicType)wrap,M,dof,s,PETSC_NULL,&da);CHKERRA(ierr);
  ierr = DACreateUniformCoordinates(da,0.0,1.0,0.0,0.0,0.0,0.0);CHKERRA(ierr);
  for ( i=0; i<dof; i++ ) {
    sprintf(fname,"Field %d",i);
    ierr = DASetFieldName(da,i,fname);
  }

  ierr = DAView(da,viewer);CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&locala);CHKERRA(ierr);
  ierr = DAGetCoordinates(da,&coors);CHKERRA(ierr);
  ierr = VecGetArray(coors,&x);CHKERRA(ierr);

  /* Set values into global vectors */
  ierr = VecGetArray(global,&alocal);CHKERRA(ierr);
  ierr = VecGetLocalSize(global,&n);CHKERRA(ierr);
  n    = n/dof;
  for ( j=0; j<dof; j++ ) {
    for ( i=0; i<n; i++ ) {
      alocal[j+dof*i] = sin(2*PETSC_PI*(j+1)*x[i]); 
    }
  }
  ierr = VecRestoreArray(global,&alocal);CHKERRA(ierr);
  ierr = VecRestoreArray(coors,&x);CHKERRA(ierr);

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
 









