#ifndef lint
static char vcid[] = "$Id: ex2.c,v 1.18 1996/03/10 17:29:57 bsmith Exp bsmith $";
#endif

/* This file was created by Peter Mell  6/30/95 */
 
static char help[] = "Tests various 1-dimensional DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>

int main(int argc,char **argv)
{
  int      rank, M = 13, ierr, w=1, s=1, wrap=1,flg;
  DA       da;
  Viewer   viewer;
  Vec      local,global;
  Scalar   value;
  Draw     draw;

  PetscInitialize(&argc,&argv,(char*)0,help);
  ierr = ViewerDrawOpenX(MPI_COMM_WORLD,0,"",280,480,600,200,&viewer); CHKERRA(ierr);
  ierr = ViewerDrawGetDraw(viewer,&draw); CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(draw); CHKERRA(ierr);

  ierr = OptionsGetInt(PETSC_NULL,"-M",&M,&flg); CHKERRA(ierr);
  ierr = OptionsGetInt(PETSC_NULL,"-w",&w,&flg);  CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-s",&s,&flg);  CHKERRA(ierr); 
  ierr = OptionsGetInt(PETSC_NULL,"-wrap",&wrap,&flg);  CHKERRA(ierr); 

  ierr = DACreate1d(MPI_COMM_WORLD,(DAPeriodicType)wrap,M,w,s,&da); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  value = 1;
  ierr = VecSet(&value,global); CHKERRA(ierr);

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  value = rank+1;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global); CHKERRA(ierr);

  ierr = VecView(global,viewer); CHKERRA(ierr); 

  PetscPrintf(MPI_COMM_WORLD,"\nGlobal Vectors:\n");
  ierr = VecView(global,STDOUT_VIEWER_WORLD); CHKERRA(ierr); 
  PetscPrintf(MPI_COMM_WORLD,"\n\n");

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  PetscPrintf(MPI_COMM_WORLD,"\nView Local Array - Processor [%d]\n",rank);
  ierr = VecView(local,STDOUT_VIEWER_WORLD); CHKERRA(ierr); 

  ierr = DAView(da,viewer); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 









