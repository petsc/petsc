  
static char help[] = "This example tests various 2d DA routines.\n\n";

#include "petsc.h"
#include "da.h"
#include "sys.h"
#include "draw.h"
#include <math.h>
#include <sysio.h>

int main(int argc,char **argv)
{
  int      mytid,M = 10, N = 8, m = PETSC_DECIDE, n = PETSC_DECIDE, ierr;
  int      s=2, w=7, xwrap=1, ywrap=1;
  DA       da;
  DrawCtx  win;
  Vec      local,global;
  Scalar   value;
 
  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stderr,"%s",help);
  ierr = DrawOpenX(MPI_COMM_WORLD,0,"",300,0,600,600,&win); CHKERRA(ierr);
 
  OptionsGetInt(0,"-M",&M);
  OptionsGetInt(0,"-N",&N);
  OptionsGetInt(0,"-m",&m);
  OptionsGetInt(0,"-n",&n);
  OptionsGetInt(0,"-s",&s);
  OptionsGetInt(0,"-w",&w);
  OptionsGetInt(0,"-xwrap",&xwrap);
  OptionsGetInt(0,"-ywrap",&ywrap);

  ierr = DACreate2dn(MPI_COMM_WORLD,M,N,m,n,w,s,xwrap,ywrap,&da); CHKERRA(ierr);
  ierr = DAGetDistributedVector(da,&global); CHKERRA(ierr);
  ierr = DAGetLocalVector(da,&local); CHKERRA(ierr);

  MPI_Comm_rank(MPI_COMM_WORLD,&mytid);
  value = 1;
  ierr = VecSet(&value,global); CHKERRA(ierr);

  ierr = DAGlobalToLocalBegin(da,global,INSERTVALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERTVALUES,local); CHKERRA(ierr);

  value = mytid;
  ierr = VecScale(&value,local); CHKERRA(ierr);
  ierr = DALocalToGlobal(da,local,INSERTVALUES,global); CHKERRA(ierr);

  if (mytid == 0) printf ("\nGlobal Vectors:\n");
  ierr = VecView(global,SYNC_STDOUT_VIEWER); CHKERRA(ierr); 
  if (mytid == 0) printf ("\n\n");

  ierr = DAGlobalToLocalBegin(da,global,INSERTVALUES,local); CHKERRA(ierr);
  ierr = DAGlobalToLocalEnd(da,global,INSERTVALUES,local); CHKERRA(ierr);

  printf ("\nView Local Array - Processor [%d]\n",mytid);
  ierr = VecView(local,SYNC_STDOUT_VIEWER); CHKERRA(ierr); 

  DAView(da,(Viewer) win);
  DADestroy(da);
  PetscFinalize();
  return 0;
}
  




















