
#include "comm.h"
#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "is.h"
#include "vec.h"

/*@C
      VecCreateInitialVector - Reads from command line to determine 
           what type of vector to create.

  Input Parameters:
.   n - total vector length
 
  Output Parameter:
.   V - location to stash resulting vector.
@*/
int VecCreateInitialVector(int n,Vec *V)
{
  if (OptionsHasName(0,"-mpi")) {
    fprintf(stdout,"Using MPI vectors\n");
    return VecCreateMPI(MPI_COMM_WORLD,-1,n,V);
  }
  if (OptionsHasName(0,"-mpiblas")) {
    fprintf(stdout,"Using MPI BLAS vectors\n");
    return VecCreateMPIBLAS(MPI_COMM_WORLD,-1,n,V);
  }
  if (OptionsHasName(0,"-blas")) {
    fprintf(stdout,"Using BLAS sequential vectors\n");
    return VecCreateSequentialBLAS(n,V);
  }
  fprintf(stdout,"Using standard sequential vectors\n");
  return VecCreateSequential(n,V);
}
 
