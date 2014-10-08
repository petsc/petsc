!
!   This program tests MatNullSpaceCreate()
!
      program main
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscis.h>
#include <finclude/petscvec.h>
#include <finclude/petscmat.h>
#include <finclude/petscviewer.h>

      PetscErrorCode ierr
      MatNullSpace nsp
      PetscScalar value
      Vec     v(1)
      PetscInt nloc

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      nloc = 12
      call VecCreateMPI(PETSC_COMM_WORLD,nloc,PETSC_DETERMINE,v(1),ierr)
      call MatNullSpaceCreate(PETSC_COMM_WORLD,PETSC_FALSE,1,v(1),         &
     &                        nsp,ierr)
      call MatNullSpaceDestroy(nsp,ierr)
      call VecDestroy(v(1),ierr)
      call PetscFinalize(ierr)
      end


