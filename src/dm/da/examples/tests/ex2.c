/*$Id: ex2.c,v 1.36 2000/05/05 22:19:31 balay Exp bsmith $*/

static char help[] = "Tests various 1-dimensional DA routines.\n\n";

#include "petscda.h"
#include "petscsys.h"

#undef __FUNC__
#define __FUNC__ "main"
int main(int argc,char **argv)
{
  int        rank,M = 13,ierr,w=1,s=1,wrap=1;
  DA         da;
  PetscViewer     viewer;
  Vec        local,global;
  Scalar     value;
  PetscDraw       draw;
  PetscTruth flg;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Create viewers */
  ierr = PetscViewerDrawOpen(PETSC_COMM_WORLD,0,"",280,480,600,200,&viewer);CHKERRA(ierr);
  ierr = PetscViewerDrawGetDraw(viewer,0,&draw);CHKERRA(ierr);
  ierr = PetscDrawSetDoubleBuffer(draw);CHKERRA(ierr);

  /* Readoptions */
  ierr = PetscOptionsGetInt(PETSC_NULL,"-M",&M,PETSC_NULL);CHKERRA(ierr);
  ierr = PetscOptionsGetInt(PETSC_NULL,"-w",&w,PETSC_NULL);CHKERRA(ierr); 
  ierr = PetscOptionsGetInt(PETSC_NULL,"-s",&s,PETSC_NULL);CHKERRA(ierr); 

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
  ierr = VecView(global,PETSC_VIEWER_STDOUT_WORLD);CHKERRA(ierr); 
  ierr = PetscPrintf(PETSC_COMM_WORLD,"\n");CHKERRA(ierr);

  /* Send ghost points to local vectors */
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local);CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local);CHKERRA(ierr);

  ierr = PetscOptionsHasName(PETSC_NULL,"-local_print",&flg);CHKERRA(ierr);
  if (flg) {
    PetscViewer sviewer;
    ierr = PetscSynchronizedPrintf(PETSC_COMM_WORLD,"\nLocal Vector: processor %d\n",rank);CHKERRA(ierr);
    ierr = PetscViewerGetSingleton(PETSC_VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
    ierr = VecView(local,sviewer);CHKERRA(ierr); 
    ierr = PetscViewerRestoreSingleton(PETSC_VIEWER_STDOUT_WORLD,&sviewer);CHKERRA(ierr);
    ierr = PetscSynchronizedFlush(PETSC_COMM_WORLD);CHKERRA(ierr);
  }

  /* Free memory */
  ierr = PetscViewerDestroy(viewer);CHKERRA(ierr);
  ierr = VecDestroy(global);CHKERRA(ierr);
  ierr = VecDestroy(local);CHKERRA(ierr);
  ierr = DADestroy(da);CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 









