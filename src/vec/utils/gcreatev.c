
#include "sys.h"
#include "sysio.h"
#include "is.h"
#include "vec.h"
#include "comm.h"

extern int VecCreateMPI(MPI_Comm,int,int,Vec *);  
extern int VecCreateMPIBLAS(MPI_Comm,int,int,Vec *);  

/*@C
      VecCreateInitialVector - Reads from command line to determine 
           what type of vector to create.

  Input Parameters:
.   argc,argv - the command line arguments
.   n - total vector length
 
  Output Parameter:
.   V - location to stash resulting vector.
@*/
int VecCreateInitialVector(int n,int argc,char **argv,Vec *V)
{
  if (SYArgHasName(&argc,argv,0,"-mpi")) {
    fprintf(stdout,"Using MPI vectors\n");
    return VecCreateMPI(MPI_COMM_WORLD,-1,n,V);
  }
#if !defined(PETSC_COMPLEX)
  if (SYArgHasName(&argc,argv,0,"-mpiblas")) {
    fprintf(stdout,"Using MPI BLAS vectors\n");
    return VecCreateMPIBLAS(MPI_COMM_WORLD,-1,n,V);
  }
  if (SYArgHasName(&argc,argv,0,"-blas")) {
    fprintf(stdout,"Using BLAS sequential vectors\n");
    return VecCreateSequentialBLAS(n,V);
  }
#endif
  fprintf(stdout,"Using standard sequential vectors\n");
  return VecCreateSequential(n,V);
}
 
