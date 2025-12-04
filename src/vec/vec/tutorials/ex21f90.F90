!
!
!    Demonstrates how one may access entries of a PETSc Vec as if it was an array of Fortran derived types
!
!
! -----------------------------------------------------------------------
#include <petsc/finclude/petscvec.h>
module ex21module
  use petscsys
  implicit none
  type MyStruct
    sequence
    PetscScalar :: a, b, c
  end type MyStruct
!
!
!   These two routines are defined in ex21.c they create the Fortran pointer to the derived type
!
  interface
    subroutine VecGetArrayMyStruct(v, array, ierr)
      use petscvec
      import MyStruct
      type(MyStruct), pointer :: array(:)
      PetscErrorCode ierr
      Vec v
    end subroutine

    subroutine VecRestoreArrayMyStruct(v, array, ierr)
      use petscvec
      import MyStruct
      type(MyStruct), pointer :: array(:)
      PetscErrorCode ierr
      Vec v
    end subroutine
  end interface
end module

!  These routines are used internally by the C functions VecGetArrayMyStruct() and VecRestoreArrayMyStruct()
!  Because Fortran requires "knowing" exactly what derived types the pointers to point too, these have to be
!  customized for exactly the derived type in question
!
subroutine F90Array1dCreateMyStruct(array, start, len, ptr)
  use petscsys
  use ex21module
  implicit none
  PetscInt start, len
  type(MyStruct), target :: array(start:start + len - 1)
  type(MyStruct), pointer :: ptr(:)

  ptr => array
end subroutine

subroutine F90Array1dAccessMyStruct(ptr, address)
  use petscsys
  use ex21module
  implicit none
  type(MyStruct), pointer :: ptr(:)
  PetscFortranAddr address
  PetscInt start

  start = lbound(ptr, 1)
  call F90Array1dGetAddrMyStruct(ptr(start), address)
end subroutine

subroutine F90Array1dDestroyMyStruct(ptr)
  use ex21module
  implicit none
  type(MyStruct), pointer :: ptr(:)

  nullify (ptr)
end subroutine

program main
  use petscvec
  use ex21module
  implicit none

!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     x, y, w - vectors
!     z       - array of vectors
!
  Vec x, y
  type(MyStruct), pointer :: xarray(:)
  PetscInt n
  PetscErrorCode ierr
  PetscBool flg
  integer i

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

  PetscCallA(PetscInitialize(ierr))
  n = 30

  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS, PETSC_NULL_CHARACTER, '-n', n, flg, ierr))
  PetscCallA(VecCreate(PETSC_COMM_WORLD, x, ierr))
  PetscCallA(VecSetSizes(x, PETSC_DECIDE, n, ierr))
  PetscCallA(VecSetFromOptions(x, ierr))
  PetscCallA(VecDuplicate(x, y, ierr))

  PetscCallA(VecGetArrayMyStruct(x, xarray, ierr))
  do i = 1, 10
    xarray(i)%a = i
    xarray(i)%b = 100*i
    xarray(i)%c = 10000*i
  end do

  PetscCallA(VecRestoreArrayMyStruct(x, xarray, ierr))
  PetscCallA(VecView(x, PETSC_VIEWER_STDOUT_SELF, ierr))
  PetscCallA(VecGetArrayMyStruct(x, xarray, ierr))
  do i = 1, 10
    write (*, *) abs(xarray(i)%a), abs(xarray(i)%b), abs(xarray(i)%c)
  end do
  PetscCallA(VecRestoreArrayMyStruct(x, xarray, ierr))

  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(VecDestroy(y, ierr))
  PetscCallA(PetscFinalize(ierr))

end

!/*TEST
!   build:
!     depends: ex21.c
!
!   test:
!
!TEST*/
