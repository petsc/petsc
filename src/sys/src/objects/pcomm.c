
#ifndef lint
static char vcid[] = "$Id: pcomm.c,v 1.1 1996/09/14 15:39:18 curfman Exp curfman $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

/*@C 
   PetscSetCommWorld - Sets a communicator to be PETSc's world communicator
   (default is MPI_COMM_WORLD).  Must call BEFORE PetscInitialize().

   Input Parameter:
.  comm - the communicator

   Note:
   This routine is intended for users who need to initialize PETSc on a
   subset of processors within a larger job.  Thus, most users need not
   call this routine.

.keywords: set, communicator, world
@*/
int PetscSetCommWorld(MPI_Comm comm)
{
  if (PetscInitializedCalled) SETERRQ(1,"PetscSetCommWorld: Must call before PetscInitialize()");
  PETSC_COMM_WORLD = comm;
  return 0;
}

