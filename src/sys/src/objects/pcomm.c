/*$Id: pcomm.c,v 1.20 2000/05/04 16:24:45 bsmith Exp bsmith $*/

#include "petsc.h"        /*I    "petsc.h"   I*/

#undef __FUNC__  
#define __FUNC__ /*<a name="PetscSetCommWorld"></a>*/"PetscSetCommWorld"
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
int PetscSetCommWorld(MPI_Comm comm)
{
  PetscFunctionBegin;
  if (PetscInitializeCalled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Must call before PetscInitialize()");
  PETSC_COMM_WORLD = comm;
  PetscFunctionReturn(0);
}

