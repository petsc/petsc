#define PETSC_DLL

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNCT__  
#define __FUNCT__ "PetscSetCommWorld"
/*@C 
   PetscSetCommWorld - Sets a communicator to be PETSc world communicator
   (default is MPI_COMM_WORLD).  Must call BEFORE PetscInitialize().

   Collective on MPI_Comm

   Input Parameter:
.  comm - the communicator

   Note:
   This routine is intended for users who need to initialize PETSc on a
   subset of processors within a larger job.  Thus, most users need not
   call this routine.

   Level: advanced

   Concepts: communicator^setting for PETSc
   Concepts: MPI communicator^setting for PETSc
   Concepts: PETSC_COMM_WORLD^setting

@*/
PetscErrorCode PETSC_DLLEXPORT PetscSetCommWorld(MPI_Comm comm)
{
  PetscFunctionBegin;
  if (PetscInitializeCalled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call before PetscInitialize()");
  PETSC_COMM_WORLD = comm;
  PetscFunctionReturn(0);
}

