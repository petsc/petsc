#ifndef lint
static char vcid[] = "$Id: ex6.c,v 1.8 1995/09/21 20:13:09 bsmith Exp bsmith $";
#endif
      
/* Peter Mell created this file on 7/25/95 */

static char help[] = 
"This example tests various 3-dimensional DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include <sysio.h>

int main(int argc,char **argv)
{
  int            mytid,M = 3, N = 5, P=3; 
  int            m = PETSC_DECIDE, n = PETSC_DECIDE, p = PETSC_DECIDE, ierr;
  int            s=1, w=2;
  DA             da;
  DrawCtx        win;
  Vec            local,global;
  Scalar         value;
  DAPeriodicType wrap = DA_XYPERIODIC;
  DAStencilType  stencil_type = DA_STENCIL_BOX;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",300,0,400,300,&win); CHKERRA(ierr);
  
  OptionsGetInt(0,"-M",&M);
  OptionsGetInt(0,"-N",&N);
  OptionsGetInt(0,"-P",&P);
  OptionsGetInt(0,"-m",&m);
  OptionsGetInt(0,"-n",&n);
  OptionsGetInt(0,"-p",&p);
  OptionsGetInt(0,"-s",&s);
  OptionsGetInt(0,"-w",&w);

  if (OptionsHasName(0,"-star")) stencil_type =  DA_STENCIL_STAR;

  ierr = DACreate3d(MPI_COMM_WORLD,wrap,stencil_type,M,N,P,m,n,p,w,s,&da); 
  CHKERRA(ierr);
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

  if (M*N*P<40)
  {
    MPIU_printf(MPI_COMM_WORLD,"\nGlobal Vectors:\n");
    ierr = VecView(global,STDOUT_VIEWER_WORLD); CHKERRA(ierr); 
    MPIU_printf(MPI_COMM_WORLD,"\n\n");
  }

  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  if (M*N*P<40) {
    MPIU_printf(MPI_COMM_WORLD,"\nView Local Array - Processor [%d]\n",mytid);
    ierr = VecView(local,STDOUT_VIEWER_WORLD); CHKERRA(ierr); 
  }
  ierr = DAView(da,(Viewer) win); CHKERRA(ierr);

  ierr = ViewerDestroy((Viewer)win); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
  




















