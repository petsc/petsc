/*$Id: ex2.c,v 1.34 1999/11/05 14:47:57 bsmith Exp bsmith $*/

static char help[] = "Tests various 1-dimensional DA routines.\n\n";

#include "da.h"
#include "sys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        rank,M = 13,ierr,w=1,s=1,wrap=1;
  DA         da;
  Viewer     viewer;
  Vec        local,global;
  Scalar     value;
  Draw       draw;
  PetscTruth flg;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Create viewers */
  ierr = ViewerDrawOpen(PETSC_COMM_WORLD,0,"",280,480,600,200,&viewer);CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(viewer,0,&draw);CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw);CHKERRA(ierr);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-w",&w,PETSC_NULL);CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRA(ierr); 

  /* Create distributed array and get vectors */
  ierr = DACreate1d(PETSC_COMM_WORLD,(DAPeriodicType)wrap,M,w,s,PETSC_NULL,&da);CHKERRA(ierr);
  ierr = DAView(da,viewer);CHKERRA(ierr);
  ierr = DACreateGlobalVector(da,&global);CHKERRA(ierr);
  ierr = DACreateLocalVector(da,&local);CHKERRA(ierr);

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  ierr = VecSet(&value,global);CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);

  /* Scale local vectors according to processor rank; pass to global vector */
  ierr = MPI_Comm_rank(PETSC_COMM_WORLD,&rank);CHKERRA(ierr);
  value = rank+1;
  ierr = VecScale(&value,local);CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global);CHKERRA(ierr);

  ierr = VecView(global,viewer);CHKERRA(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\nGlobal Vector:\n");CHKERRA(ierr);
  ierr = VecView(global,VIEWER_STDOUT_WORLD);CHKERRA(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRA(ierr);

  /* Send ghost points to local vectors */
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);

  ierr = OptionsHasName(PETSC_NULL,"-local_print",&flg);CHKERRA(ierr);
  if (flg) {
    Viewer sviewer;
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nLocal Vector: processor %d\n",rank);CHKERRA(ierr);
    ierr = ViewerGetSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
    ierr = VecView(local,sviewer);CHKERRA(ierr); 
    ierr = ViewerRestoreSingleton(VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);
  }

  /* Free memory */
  ierr = ViewerDestroy(viewer);CHKERRA(ierr);
  ierr = VecDestroy(global);CHKERRA(ierr);
  ierr = VecDestroy(local);CHKERRA(ierr);
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 









