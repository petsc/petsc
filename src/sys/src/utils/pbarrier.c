
#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: pbarrier.c,v 1.6 1999/09/21 15:14:07 bsmith Exp bsmith $";
#endif

#include "petsc.h"              /*I "petsc.h" I*/

#undef __FUNC__  
#define __FUNC__ "PetscBarrier"
/*@C
    PetscBarrier - Blocks until this routine is executed by all
                   processors owning the object A.

   Input Parameters:
.  A - PETSc object  ( Mat, Vec, IS, SNES etc...)
        Must be caste with a (PetscObject)

  Level: intermediate

  Notes: 
  This routine calls MPI_Barrier with the communicator of the PETSc Object "A". 

.keywords: barrier, petscobject

@*/
int PetscBarrier(PetscObject obj)
{
  int      ierr;
  MPI_Comm comm;

  PetscFunctionBegin;
  PetscValidHeader(obj); 
  PLogEventBegin(Petsc_Barrier,obj,0,0,0); 
  ierr = PetscObjectGetComm(obj,&comm);CHKERRQ(ierr);
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  PLogEventEnd(Petsc_Barrier,obj,0,0,0); 
  PetscFunctionReturn(0);
}

