
#ifndef lint
static char vcid[] = "$Id: pcomm.c,v 1.1 1996/09/14 03:05:57 bsmith Exp $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

/*@C 
   PetscSetCommWorld - Sets a communicator that is PETSc's world communicator
   (default is MPI_COMM_WORLD).  Must call BEFORE PETScInitialize().

   Input Parameter:
.  comm - the communicator

.keywords: set, communicator, world
@*/
int PetscSetCommWorld(MPI_Comm comm)
{
  if (PetscInitializedCalled) SETERRQ(1,"PetscSetCommWorld: Must call before PetscInitialize()");
  PETSC_COMM_WORLD = comm;
  return 0;
}

