#ifndef lint
static char vcid[] = "$Id: ex4.c,v 1.8 1995/08/30 01:27:14 curfman Exp bsmith $";
#endif
  
static char help[] = 
"This example tests various 2-dimensional DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include <sysio.h>

int main(int argc,char **argv)
{
  int            mytid,M = 10, N = 8, m = PETSC_DECIDE,ierr;
  int            s=2, w=2,n = PETSC_DECIDE ;
  DAPeriodicType wrap = DA_NONPERIODIC;
  DA             da;
  DrawCtx        win;
  Vec            local,global;
  Scalar         value;
  DAStencilType  st = DA_STENCIL_BOX;
 
  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",300,0,400,400,&win);
  CHKERRA(ierr);
 
  OptionsGetInt(0,"-M",&M);
  OptionsGetInt(0,"-N",&N);
  OptionsGetInt(0,"-m",&m);
  OptionsGetInt(0,"-n",&n);
  OptionsGetInt(0,"-s",&s);
  OptionsGetInt(0,"-w",&w);
  if (OptionsHasName(0,"-xwrap")) wrap = DA_XPERIODIC;
  if (OptionsHasName(0,"-ywrap")) wrap = DA_YPERIODIC;
  if (OptionsHasName(0,"-xywrap")) wrap = DA_XYPERIODIC;
  if (OptionsHasName(0,"-star")) st = DA_STENCIL_STAR;

  ierr = DACreate2d(MPI_COMM_WORLD,wrap,st,M,N,m,n,w,s,&da); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  value = 1;
  ierr = VecSet(&value,global); CHKERRA(ierr);

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  value = mytid;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global); CHKERRA(ierr);

  MPIU_printf (MPI_COMM_WORLD,"\nGlobal Vectors:\n");
  ierr = VecView(global,STDOUT_VIEWER_WORLD); CHKERRA(ierr); 
  MPIU_printf (MPI_COMM_WORLD,"\n\n");

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  MPIU_printf (MPI_COMM_WORLD,"\nView Local Array - Processor [%d]\n",mytid);

  ierr = DAView(da,(Viewer) win); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  ierr = ViewerDestroy((Viewer)win); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
  




















