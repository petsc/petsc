
static char help[] = "This example tests various DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include <sysio.h>

int main(int argc,char **argv)
{
  int      mytid,M = 10, N = 8, m = PETSC_DECIDE, n = PETSC_DECIDE, ierr;
  DA       da;
  DrawCtx  win;
  Vec      local,global;
  Scalar   value;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",300,0,300,300,&win); CHKERRA(ierr);

  OptionsGetInt(0,"-M",&M);
  OptionsGetInt(0,"-N",&N);
  OptionsGetInt(0,"-m",&m);
  OptionsGetInt(0,"-n",&n);

  ierr = DACreate2d(MPI_COMM_WORLD,DA_NONPERIODIC,DA_STENCIL_BOX,
                    M,N,m,n,1,1,&da); 
  CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  value = -3.0;
  ierr = VecSet(&value,global); CHKERRA(ierr);
  ierr = DAGlobalToLocalBegin(da,global,INSERTVALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERTVALUES,local); CHKERRA(ierr);

  /* ierr = VecView(local,SYNC_STDOUT_VIEWER); CHKERRA(ierr); */

  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  value = mytid+1;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,ADDVALUES,global); CHKERRA(ierr);

  ierr = VecView(global,SYNC_STDOUT_VIEWER); CHKERRA(ierr);
  


  DAView(da,(Viewer) win);
  DADestroy(da);
  PetscFinalize();
  return 0;
}
 
