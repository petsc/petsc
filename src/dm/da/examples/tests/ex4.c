#ifndef lint
static char vcid[] = "$Id: ex4.c,v 1.15 1995/11/30 22:36:31 bsmith Exp bsmith $";
#endif
  
static char help[] = "Tests various 2-dimensional DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include "sysio.h"

int main(int argc,char **argv)
{
  int            rank,M = 10, N = 8, m = PETSC_DECIDE,ierr;
  int            s=2, w=2,n = PETSC_DECIDE ;
  DAPeriodicType wrap = DA_NONPERIODIC;
  DA             da;
  Draw        win;
  Vec            local,global;
  Scalar         value;
  DAStencilType  st = DA_STENCIL_BOX;
 
  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",300,0,400,400,&win);
  CHKERRA(ierr);
 
  OptionsGetInt(PETSC_NULL,"-M",&M);
  OptionsGetInt(PETSC_NULL,"-N",&N);
  OptionsGetInt(PETSC_NULL,"-m",&m);
  OptionsGetInt(PETSC_NULL,"-n",&n);
  OptionsGetInt(PETSC_NULL,"-s",&s);
  OptionsGetInt(PETSC_NULL,"-w",&w);
  if (OptionsHasName(PETSC_NULL,"-xwrap")) wrap = DA_XPERIODIC;
  if (OptionsHasName(PETSC_NULL,"-ywrap")) wrap = DA_YPERIODIC;
  if (OptionsHasName(PETSC_NULL,"-xywrap")) wrap = DA_XYPERIODIC;
  if (OptionsHasName(PETSC_NULL,"-star")) st = DA_STENCIL_STAR;

  ierr = DACreate2d(MPI_COMM_WORLD,wrap,st,M,N,m,n,w,s,&da); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  value = 1;
  ierr = VecSet(&value,global); CHKERRA(ierr);

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  value = rank;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERT_VALUES,global); CHKERRA(ierr);

  MPIU_printf (MPI_COMM_WORLD,"\nGlobal Vectors:\n");
  ierr = VecView(global,STDOUT_VIEWER_WORLD); CHKERRA(ierr); 
  MPIU_printf (MPI_COMM_WORLD,"\n\n");

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  MPIU_printf (MPI_COMM_WORLD,"\nView Local Array - Processor [%d]\n",rank);

  ierr = DAView(da,(Viewer) win); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  ierr = ViewerDestroy((Viewer)win); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
  




















