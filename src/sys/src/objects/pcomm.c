
#ifndef lint
static char vcid[] = "$Id: pcomm.c,v 1.3 1996/12/16 21:56:21 balay Exp balay $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNCTION__  
#define __FUNCTION__ "PetscSetCommWorld"
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
  if (PetscInitializedCalled) SETERRQ(1,"Must call before PetscInitialize()");
  PETSC_COMM_WORLD = comm;
  return 0;
}

