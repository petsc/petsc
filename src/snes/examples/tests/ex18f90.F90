!
! Example usage of Fortran 2003/2008 classes (extended derived types) as
! user-defined contexts in PETSc. Example contributed by Glenn Hammond.
!
module Base_module

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
end module Base_module

module Extended_module
  use Base_module
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
end module Extended_module

module Function_module
  use petscsnes
  implicit none
  public :: TestFunction
  contains
subroutine TestFunction(snes,xx,r,ctx,ierr)
  use Base_module
  implicit none
  SNES :: snes
  Vec :: xx
  Vec :: r
  class(base_type) :: ctx ! yes, this should be base_type in order to handle all
  PetscErrorCode :: ierr  ! polymorphic extensions
  call ctx%Print()
end subroutine TestFunction
end module Function_module

program ex18f90

  use Base_module
  use Extended_module
  use Function_module
  implicit none

! ifort on windows requires this interface definition
interface
  subroutine SNESSetFunction(snes_base,x,TestFunction,base,ierr)
    use Base_module
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
  call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
  if (ierr .ne. 0) then
    print*,'Unable to initialize PETSc'
    stop
  endif
  call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr);CHKERRA(ierr)
  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);CHKERRA(ierr)

  call VecCreate(PETSC_COMM_WORLD,x,ierr);CHKERRA(ierr)

  ! use the base class as the context
  print *
  print *, 'the base class will succeed by printing out Base printout below'
  call SNESCreate(PETSC_COMM_WORLD,snes_base,ierr);CHKERRA(ierr)
  call SNESSetFunction(snes_base,x,TestFunction,base,ierr);CHKERRA(ierr)
  call SNESComputeFunction(snes_base,x,x,ierr);CHKERRA(ierr)
  call SNESDestroy(snes_base,ierr);CHKERRA(ierr)

  ! use the extended class as the context
  print *, 'the extended class will succeed by printing out Extended printout below'
  call SNESCreate(PETSC_COMM_WORLD,snes_extended,ierr);CHKERRA(ierr)
  call SNESSetFunction(snes_extended,x,TestFunction,extended,ierr);CHKERRA(ierr)
  call SNESComputeFunction(snes_extended,x,x,ierr);CHKERRA(ierr)
  call VecDestroy(x,ierr);CHKERRA(ierr)
  call SNESDestroy(snes_extended,ierr);CHKERRA(ierr)
  if (associated(base)) deallocate(base)
  if (associated(extended)) deallocate(extended)
  call PetscFinalize(ierr)

  print *, 'End of Fortran2003 test program'

end program ex18f90

!/*TEST
!
!   build:
!      requires: define(PETSC_USING_F2003) define(PETSC_USING_F90FREEFORM)
!   test:
!     requires: !pgf90_compiler
!
!TEST*/
