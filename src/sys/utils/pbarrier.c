
#include <petscsys.h>              /*I "petscsys.h" I*/

/* Logging support */
PetscLogEvent PETSC_Barrier=0;

#undef __FUNCT__
#define __FUNCT__ "PetscBarrier"
/*@C
    PetscBarrier - Blocks until this routine is executed by all
                   processors owning the object A.

   Input Parameters:
.  A - PETSc object  (Mat, Vec, IS, SNES etc...)
        Must be caste with a (PetscObject), can use PETSC_NULL (for MPI_COMM_WORLD)

  Level: intermediate

  Notes:
  This routine calls MPI_Barrier with the communicator of the PETSc Object "A".

  With fortran Use PETSC_NULL_OBJECT (instead of PETSC_NULL)

   Concepts: barrier

@*/
PetscErrorCode  PetscBarrier(PetscObject obj)
{
  PetscErrorCode ierr;
  MPI_Comm       comm;

  PetscFunctionBegin;
  if (obj) PetscValidHeader(obj,1);
  ierr = PetscLogEventBegin(PETSC_Barrier,obj,0,0,0);CHKERRQ(ierr);
  if (obj) {
    ierr = PetscObjectGetComm(obj,&comm);CHKERRQ(ierr);
  } else {
    comm = PETSC_COMM_WORLD;
  }
  ierr = MPI_Barrier(comm);CHKERRQ(ierr);
  ierr = PetscLogEventEnd(PETSC_Barrier,obj,0,0,0);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

