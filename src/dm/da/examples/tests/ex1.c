
static char help[] = "This example tests various RA routines.\n\n";

#include "petsc.h"
#include "ra.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include <sysio.h>

int main(int argc,char **argv)
{
  int      mytid,M = 10, N = 8, m = PETSC_DECIDE, n = PETSC_DECIDE, ierr;
  RA       ra;
  DrawCtx  win;
  Vec      local,global;
  Scalar   value;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",300,0,300,300,&win); CHKERRA(ierr);

  OptionsGetInt(0,0,"-M",&M);
  OptionsGetInt(0,0,"-N",&N);
  OptionsGetInt(0,0,"-m",&m);
  OptionsGetInt(0,0,"-n",&n);

  ierr = RACreate2d(MPI_COMM_WORLD,M,N,m,n,1,1,&ra); CHKERRA(ierr);
  ierr = RAGetDistributedVector(ra,&global); CHKERR(ierr);
  ierr = RAGetLocalVector(ra,&local); CHKERR(ierr);

  value = -3.0;
  ierr = VecSet(&value,global); CHKERR(ierr);
  ierr = RAGlobalToLocal(ra,global,INSERTVALUES,local); CHKERR(ierr);

  /* ierr = VecView(local,SYNC_STDOUT_VIEWER); CHKERR(ierr); */

  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  value = mytid+1;
  ierr = VecScale(&value,local); CHKERR(ierr);
  ierr = RALocalToGlobal(ra,local,ADDVALUES,global); CHKERR(ierr);

  ierr = VecView(global,SYNC_STDOUT_VIEWER); CHKERR(ierr);
  


  RAView(ra,(Viewer) win);
  RADestroy(ra);
  PetscFinalize();
  return 0;
}
 
