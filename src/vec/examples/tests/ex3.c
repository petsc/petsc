#ifndef lint
static char vcid[] = "$Id: ex3.c,v 1.22 1995/08/17 21:33:33 curfman Exp curfman $";
#endif

static char help[] = "This example tests parallel vector assembly.  Input\n\
arguments are\n\
  -n <length> : local vector length\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int          n = 5, ierr, numtids,mytid;
  Scalar       one = 1.0, two = 2.0, three = 3.0;
  Vec          x,y;
  int          idx;

  PetscInitialize(&argc,&argv,(char*)0,(char*)0);
  if (OptionsHasName(0,"-help")) fprintf(stdout,"%s",help);
  OptionsGetInt(0,"-n",&n); if (n < 5) n = 5;
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid); 

  if (numtids < 2) SETERRA(1,"Must be run with at least two processors");

  /* create two vector */
  ierr = VecCreateSequential(MPI_COMM_SELF,n,&x); CHKERRA(ierr);
  ierr = VecCreateMPI(MPI_COMM_WORLD,n,PETSC_DECIDE,&y); CHKERRA(ierr);
  ierr = VecSet(&one,x); CHKERRA(ierr);
  ierr = VecSet(&two,y); CHKERRA(ierr);

  if (mytid == 1) {
    idx = 2; ierr = VecSetValues(y,1,&idx,&three,INSERTVALUES); CHKERRA(ierr);
    idx = 0; ierr = VecSetValues(y,1,&idx,&two,INSERTVALUES); CHKERRA(ierr); 
    idx = 0; ierr = VecSetValues(y,1,&idx,&one,INSERTVALUES); CHKERRA(ierr); 
  }
  else {
    idx = 7; ierr = VecSetValues(y,1,&idx,&three,INSERTVALUES); CHKERRA(ierr); 
  } 
  ierr = VecAssemblyBegin(y); CHKERRA(ierr);
  ierr = VecAssemblyEnd(y); CHKERRA(ierr);

  ierr = VecView(y,STDOUT_VIEWER_COMM); CHKERRA(ierr);

  ierr = VecDestroy(x); CHKERRA(ierr);
  ierr = VecDestroy(y); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
