#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pcomm.c,v 1.13 1998/04/27 19:48:45 curfman Exp bsmith $";
#endif

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ "PetscSetCommWorld"
/*@C 
   PetscSetCommWorld - Sets a communicator to be PETSc's world communicator
   (default is MPI_COMM_WORLD).  Must call BEFORE PetscInitialize().

   Collective on MPI_Comm

   Input Parameter:
.  comm - the communicator

   Note:
   This routine is intended for users who need to initialize PETSc on a
   subset of processors within a larger job.  Thus, most users need not
   call this routine.

   Level: advanced

.keywords: set, communicator, world
@*/
int PetscSetCommWorld(MPI_Comm comm)
{
  PetscFunctionBegin;
  if (PetscInitializedCalled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Must call before PetscInitialize()");
  PETSC_COMM_WORLD = comm;
  PetscFunctionReturn(0);
}

