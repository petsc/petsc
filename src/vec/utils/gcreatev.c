#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.11 1995/04/17 19:58:12 bsmith Exp curfman $";
#endif


#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "is.h"
#include "vec.h"

/*@C
    VecCreateInitialVector - Creates a vector, reading from the command
    line to determine the vector type.  Generates a parallel MPI vector
    if the communicator has more than one processor.

    Input Parameters:
.   comm - MPI communicator
.   n - global vector length
 
    Output Parameter:
.   V - location to stash resulting vector

    Options Database Key:
$   -mpi_objects : use MPI vectors, even for the uniprocessor case

.keywords: vector, create, initial

.seealso: VecCreateSequential(), VecCreateMPI()
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
 
