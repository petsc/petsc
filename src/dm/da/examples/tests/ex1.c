#ifndef lint
static char vcid[] = "$Id: ex1.c,v 1.20 1995/11/09 22:33:23 bsmith Exp bsmith $";
#endif

static char help[] = "Tests various DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include "sysio.h"

int main(int argc,char **argv)
{
  int      rank, M = 10, N = 8, m = PETSC_DECIDE, n = PETSC_DECIDE, ierr;
  DA       da;
  Draw  win;
  Vec      local, global;
  Scalar   value;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0,help);
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",300,0,300,300,&win); CHKERRA(ierr);

  OptionsGetInt(PetscNull,"-M",&M);
  OptionsGetInt(PetscNull,"-N",&N);
  OptionsGetInt(PetscNull,"-m",&m);
  OptionsGetInt(PetscNull,"-n",&n);

  ierr = DACreate2d(MPI_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,
                    M,N,m,n,1,1,&da); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  value = -3.0;
  ierr = VecSet(&value,global); CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERT_VALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERT_VALUES,local); CHKERRA(ierr);

  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  value = rank+1;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,ADD_VALUES,global); CHKERRA(ierr);

  ierr = VecView(global,STDOUT_VIEWER_WORLD); CHKERRA(ierr);
  ierr = DAView(da,(Viewer) win); CHKERRA(ierr);

  ierr = ViewerDestroy((Viewer)win); CHKERRA(ierr);
  ierr = VecDestroy(local); CHKERRA(ierr);
  ierr = VecDestroy(global); CHKERRA(ierr);
  ierr = DADestroy(da); CHKERRA(ierr);
  PetscFinalize();
  return 0;
}
 
