#ifndef lint
static char vcid[] = "$Id: gcreate.c,v 1.12 1995/04/27 01:00:23 curfman Exp curfman $";
#endif


#include "sys.h"
#include "options.h"
#include "sysio.h"
#include "is.h"
#include "vec.h"

/*@C
    VecCreateInitialVector - Creates a vector, where the vector type 
    is determined from the options database.  Generates a parallel MPI 
    vector if the communicator has more than one processor.

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
 
