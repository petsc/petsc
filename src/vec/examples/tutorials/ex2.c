

static char help[] = "Builds a parallel vector with 1 component on the first\n\
   processor 2 on the second,... Then each processor adds one to all elements\n\
   except the last mytid.\n";

#include "petsc.h"
#include "is.h"
#include "vec.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include <math.h>

int main(int argc,char **argv)
{
  int          i,N,ierr, numtids,mytid;
  Scalar       one = 1.0;
  Vec          x;

  PetscInitialize(&argc,&argv,0,0);
  if (OptionsHasName(0,0,"-help")) fprintf(stderr,"%s",help);
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid); 

  ierr = VecCreateMPI(MPI_COMM_WORLD,mytid+1,PETSC_DECIDE,&x); CHKERRA(ierr);
  ierr = VecGetSize(x,&N);
  ierr = VecSet(&one,x);CHKERRA(ierr);

  for ( i=0; i<N-mytid; i++ ) {
    ierr = VecSetValues(x,1,&i,&one,AddValues); CHKERRA(ierr);  
  }
  ierr = VecAssemblyBegin(x); CHKERRA(ierr);
  ierr = VecAssemblyEnd(x); CHKERRA(ierr);

  VecView(x,SYNC_STDOUT_VIEWER);

  ierr = VecDestroy(x);CHKERRA(ierr);

  PetscFinalize();
  return 0;
}
 
