#ifndef lint
static char vcid[] = "$Id: ex4.c,v 1.14 1995/11/09 22:33:23 bsmith Exp bsmith $";
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
 
  OptionsGetInt(PetscNull,"-M",&M);
  OptionsGetInt(PetscNull,"-N",&N);
  OptionsGetInt(PetscNull,"-m",&m);
  OptionsGetInt(PetscNull,"-n",&n);
  OptionsGetInt(PetscNull,"-s",&s);
  OptionsGetInt(PetscNull,"-w",&w);
  if (OptionsHasName(PetscNull,"-xwrap")) wrap = DA_XPERIODIC;
  if (OptionsHasName(PetscNull,"-ywrap")) wrap = DA_YPERIODIC;
  if (OptionsHasName(PetscNull,"-xywrap")) wrap = DA_XYPERIODIC;
  if (OptionsHasName(PetscNull,"-star")) st = DA_STENCIL_STAR;

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
  




















