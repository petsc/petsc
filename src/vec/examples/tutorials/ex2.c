#ifndef lint
static char vcid[] = "$Id: ex10.c,v 1.19 1995/09/30 19:26:45 bsmith Exp bsmith $";
#endif

static char help[] = "Builds a parallel vector with 1 component on the first\n\
processor, 2 on the second, etc.  Then each processor adds one to all\n\
elements except the last mytid.\n\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int          i,N,ierr, numtids,mytid;
  Scalar       one = 1.0;
  Vec          x;

  PetscInitialize(&argc,&argv,0,0,help);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid); 

  ierr = VecCreateMPI(MPI_COMM_WORLD,mytid+1,PETSC_DECIDE,&x); CHKERRA(ierr);
  ierr = VecGetSize(x,&N); CHKERRA(ierr);
  ierr = VecSet(&one,x); CHKERRA(ierr);

  for ( i=0; i<N-mytid; i++ ) {
    ierr = VecSetValues(x,1,&i,&one,ADD_VALUES); CHKERRA(ierr);  
  }
  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);

  ierr = VecView(x,STDOUT_VIEWER_WORLD); CHKERRA(ierr);
  ierr = VecDestroy(x); CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
