#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.10 1995/04/15 03:26:19 bsmith Exp bsmith $";
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
.   comm - MPI communicator
.   n - total vector length
 
  Output Parameter:
.   V - location to stash resulting vector.
@*/
int VecCreateInitialVector(MPI_Comm comm,int n,Vec *V)
{
  int numtid;
  MPI_Comm_size(comm,&numtid);
  if (numtid > 1 || OptionsHasName(0,0,"-mpi_objects")) {
    return VecCreateMPI(comm,PETSC_DECIDE,n,V);
  }
  return VecCreateSequential(comm,n,V);
}
 
