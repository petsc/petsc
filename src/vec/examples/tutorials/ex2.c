

/*
      Builds a parallel vector with 1 component on the first processor
   2 on the second,... Then each processor adds one to all elements 
   except the last mytid.

*/
#include "petsc.h"
#include "comm.h"
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
  MPI_Comm_size(MPI_COMM_WORLD,&numtids);
  MPI_Comm_rank(MPI_COMM_WORLD,&mytid); 

  ierr = VecCreateMPI(MPI_COMM_WORLD,mytid+1,-1,&x); CHKERR(ierr);
  ierr = VecGetSize(x,&N);
  ierr = VecSet(&one,x);CHKERR(ierr);

  for ( i=0; i<N-mytid; i++ ) {
    ierr = VecAddValues(x,1,&i,&one); CHKERR(ierr);  
  }
  ierr = VecBeginAssembly(x); CHKERR(ierr);
  ierr = VecEndAssembly(x); CHKERR(ierr);

  VecView(x,0);

  ierr = VecDestroy(x);CHKERR(ierr);

  PetscFinalize();
  return 0;
}
 
