!
! Example usage of Fortran 2003/2008 classes (extended derived types) as
! user-defined contexts in PETSc. Example contributed by Glenn Hammond.
!
module ex18f90base_module
#include "petsc/finclude/petscsnes.h"
      implicit none
  private

  type, public :: base_type
    PetscInt :: A  ! junk
    PetscReal :: I ! junk
  contains
    procedure, public :: Print => BasePrint
  end type base_type
contains
subroutine BasePrint(this)
  implicit none
  class(base_type) :: this
  print *
  print *, 'Base printout'
  print *
end subroutine BasePrint
end module ex18f90base_module

module ex18f90extended_module
  use ex18f90base_module
  implicit none
  private
  type, public, extends(base_type) :: extended_type
    PetscInt :: B  ! junk
    PetscReal :: J ! junk
  contains
    procedure, public :: Print =>  ExtendedPrint
  end type extended_type
contains
subroutine ExtendedPrint(this)
  implicit none
  class(extended_type) :: this
  print *
  print *, 'Extended printout'
  print *
end subroutine ExtendedPrint
end module ex18f90extended_module

module ex18f90function_module
  use petscsnes
  implicit none
  public :: TestFunction
  contains
subroutine TestFunction(snes,xx,r,ctx,ierr)
  use ex18f90base_module
  implicit none
  SNES :: snes
  Vec :: xx
  Vec :: r
  class(base_type) :: ctx ! yes, this should be base_type in order to handle all
  PetscErrorCode :: ierr  ! polymorphic extensions
  call ctx%Print()
end subroutine TestFunction
end module ex18f90function_module

program ex18f90

  use ex18f90base_module
  use ex18f90extended_module
  use ex18f90function_module
  implicit none

! ifort on windows requires this interface definition
interface
  subroutine SNESSetFunction(snes_base,x,TestFunction,base,ierr)
    use ex18f90base_module
    use petscsnes
    SNES snes_base
    Vec x
    external TestFunction
    class(base_type) :: base
    PetscErrorCode ierr
  end subroutine
end interface

  PetscMPIInt :: size
  PetscMPIInt :: rank

  SNES :: snes_base, snes_extended
  Vec :: x
  class(base_type), pointer :: base
  class(extended_type), pointer :: extended
  PetscErrorCode :: ierr

  print *, 'Start of Fortran2003 test program'

  nullify(base)
  nullify(extended)
  allocate(base)
  allocate(extended)
  PetscCallA(PetscInitialize(ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

  PetscCallA(VecCreate(PETSC_COMM_WORLD,x,ierr))

  ! use the base class as the context
  print *
  print *, 'the base class will succeed by printing out Base printout below'
  PetscCallA(SNESCreate(PETSC_COMM_WORLD,snes_base,ierr))
  PetscCallA(SNESSetFunction(snes_base,x,TestFunction,base,ierr))
  PetscCallA(SNESComputeFunction(snes_base,x,x,ierr))
  PetscCallA(SNESDestroy(snes_base,ierr))

  ! use the extended class as the context
  print *, 'the extended class will succeed by printing out Extended printout below'
  PetscCallA(SNESCreate(PETSC_COMM_WORLD,snes_extended,ierr))
  PetscCallA(SNESSetFunction(snes_extended,x,TestFunction,extended,ierr))
  PetscCallA(SNESComputeFunction(snes_extended,x,x,ierr))
  PetscCallA(VecDestroy(x,ierr))
  PetscCallA(SNESDestroy(snes_extended,ierr))
  if (associated(base)) deallocate(base)
  if (associated(extended)) deallocate(extended)
  PetscCallA(PetscFinalize(ierr))

  print *, 'End of Fortran2003 test program'
end program ex18f90

!/*TEST
!
!   build:
!      requires: defined(PETSC_USING_F2003) defined(PETSC_USING_F90FREEFORM)
!   test:
!     requires: !pgf90_compiler
!
!TEST*/
