#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.8 1995/03/06 04:01:07 bsmith Exp bsmith $";
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
  if (OptionsHasName(0,0,"-mpi_blas_objects")) {
    return VecCreateMPIBLAS(MPI_COMM_WORLD,-1,n,V);
  }
  if (numtid > 1 || OptionsHasName(0,0,"-mpi_objects")) {
    return VecCreateMPI(MPI_COMM_WORLD,-1,n,V);
  }
  if (OptionsHasName(0,0,"-blas_objects")) {
    return VecCreateSequentialBLAS(n,V);
  }
  return VecCreateSequential(n,V);
}
 
