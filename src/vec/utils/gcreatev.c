#ifndef lint
static char vcid[] = "$Id: $";
#endif


#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "is.h"
#include "vec.h"

/*@C
      VecCreateInitialVector - Reads from command line to determine 
           what type of vector to create. Also generates parallel vector
           if MPI_COMM_WORLD has more then one processor.

  Input Parameters:
.   n - total vector length
 
  Output Parameter:
.   V - location to stash resulting vector.
@*/
int VecCreateInitialVector(int n,Vec *V)
{
  int numtid;
  MPI_Comm_size(MPI_COMM_WORLD,&numtid);
  if (OptionsHasName(0,0,"-mpi_blas_vecs")) {
    return VecCreateMPIBLAS(MPI_COMM_WORLD,-1,n,V);
  }
  if (OptionsHasName(0,0,"-mpi_vecs") || numtid > 1) {
    return VecCreateMPI(MPI_COMM_WORLD,-1,n,V);
  }
  if (OptionsHasName(0,0,"-blas_vecs")) {
    return VecCreateSequentialBLAS(n,V);
  }
  return VecCreateSequential(n,V);
}
 
