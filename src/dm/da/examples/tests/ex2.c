
/* This file was created by Peter Mell  6/30/95 */
 
static char help[] = 
"This example tests various 1-dimensional DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include <sysio.h>

int main(int argc,char **argv)
{
  int      mytid, M = 13, ierr, w=1, s=1, wrap=1;
  DA       da;
  DrawCtx  win1,win2;
  Vec      local,global;
  Scalar   value;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",280,480,600,200,&win1); CHKERRA(ierr);
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",280,258,600,200,&win2); CHKERRA(ierr);
  ierr = DrawSetDoubleBuffer(win1); CHKERRA(ierr);

  OptionsGetInt(0,"-M",&M);
  OptionsGetInt(0,"-w",&w);  /* degrees of freedom */ 
  OptionsGetInt(0,"-s",&s);  /* stencil width */
  OptionsGetInt(0,"-wrap",&wrap);  /* wrap or not */

  ierr = DACreate1d(MPI_COMM_WORLD,(DAPeriodicType)wrap,M,w,s,&da); 
  CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  value = 1;
  ierr = VecSet(&value,global); CHKERRA(ierr);

  ierr = DAGlobalToLocalBegin(da,global,INSERTVALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERTVALUES,local); CHKERRA(ierr);

  value = mytid+1;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERTVALUES,global); CHKERRA(ierr);

  ierr = VecView(global,(Viewer) win1); CHKERRA(ierr); 

  MPIU_printf(MPI_COMM_WORLD,"\nGlobal Vectors:\n");
  ierr = VecView(global,SYNC_STDOUT_VIEWER); CHKERRA(ierr); 
  MPIU_printf(MPI_COMM_WORLD,"\n\n");

  ierr = DAGlobalToLocalBegin(da,global,INSERTVALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERTVALUES,local); CHKERRA(ierr);

  MPIU_printf(MPI_COMM_WORLD,"\nView Local Array - Processor [%d]\n",mytid);
  ierr = VecView(local,SYNC_STDOUT_VIEWER); CHKERRA(ierr); 

  ierr = DAView(da,(Viewer) win2); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 









