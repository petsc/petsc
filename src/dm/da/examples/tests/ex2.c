#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.20 1996/03/23 00:34:08 curfman Exp bsmith $";
#endif

static char help[] = "Tests various 1-dimensional DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  int    rank, M = 13, ierr, w=1, s=1, wrap=1, flg;
  DA     da;
  Viewer viewer;
  Vec    local, global;
  Scalar value;
  Draw   draw;

  PetscInitialize(&argc,&argv,(char*)0,help);

  /* Create viewers */
  ierr = ViewerDrawOpenX(MPI_COMM_WORLD,0,"",280,480,600,200,&viewer); CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw); CHKERRA(ierr);

  /* Read options */
  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-w",&w,&flg);  CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,&flg);  CHKERRA(ierr); 

  /* Create distributed array and get vectors */
  ierr = DACreate1d(MPI_COMM_WORLD,(DAPeriodicType)wrap,M,w,s,&da); CHKERRA(ierr);
  ierr = DAView(da,viewer); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  /* Set global vector; send ghost points to local vectors */
  value = 1;
  ierr = VecSet(&value,global); CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  /* Scale local vectors according to processor rank; pass to global vector */
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  value = rank+1;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global); CHKERRA(ierr);

  ierr = VecView(global,viewer); CHKERRA(ierr); 
  PetscPrintf(MPI_COMM_WORLD,"\nGlobal Vector:\n");
  ierr = VecView(global,VIEWER_STDOUT_WORLD); CHKERRA(ierr); 
  PetscPrintf(MPI_COMM_WORLD,"\n");

  /* Send ghost points to local vectors */
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  flg = 0;
  ierr = OptionsHasName(PETSC_NULL,"-local_print",&flg); CHKERRA(ierr);
  if (flg) {
    PetscSequentialPhaseBegin(MPI_COMM_WORLD,1);
    printf("\nLocal Vector: processor %d\n",rank);
    ierr = VecView(local,VIEWER_STDOUT_SELF); CHKERRA(ierr); 
    PetscSequentialPhaseEnd(MPI_COMM_WORLD,1);
  }

  /* Free memory */
  ierr = ViewerDestroy(viewer); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 









