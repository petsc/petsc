!
!   Demonstrates PetscObjectNullify()
!
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

      PetscErrorCode ierr
      Vec x, y
      PetscInt :: one = 1

      PetscCallA(PetscInitialize(ierr))

      PetscCallA(VecCreateMPI(PETSC_COMM_WORLD, one, PETSC_DETERMINE, x, ierr))
      y = x
      PetscCallA(VecDestroy(x, ierr))
      if (.not. PetscObjectIsNull(y)) then
        print *, "y appears to be a valid object when it is not because x has been destroyed"
      end if
      PetscObjectNullify(y)
      !  Using y = PETSC_NULL_VEC would cause a crash in VecCreateMPI()
      PetscCallA(VecCreateMPI(PETSC_COMM_WORLD, one, PETSC_DETERMINE, y, ierr))
      PetscCallA(VecDestroy(y, ierr))
      PetscCallA(PetscFinalize(ierr))
    end

!/*TEST
!
!     test:
!       nsize: 1
!
!TEST*/
