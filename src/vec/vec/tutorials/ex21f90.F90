!
!
!    Demonstrates how one may access entries of a PETSc Vec as if it was an array of Fortran derived types
!
!/*T
!   Concepts: vectors^basic routines;
!   Processors: n
!T*/
!
! -----------------------------------------------------------------------

      module mymoduleex21f90
#include <petsc/finclude/petscsys.h>
      type MyStruct
        sequence
        PetscScalar :: a,b,c
      end type MyStruct
      end module

!
!  These routines are used internally by the C functions VecGetArrayMyStruct() and VecRestoreArrayMyStruct()
!  Because Fortran requires "knowing" exactly what derived types the pointers to point too, these have to be
!  customized for exactly the derived type in question
!
      subroutine F90Array1dCreateMyStruct(array,start,len,ptr)
#include <petsc/finclude/petscsys.h>
      use petscsys
      use mymoduleex21f90
      implicit none
      PetscInt start,len
      type(MyStruct), target :: array(start:start+len-1)
      type(MyStruct), pointer :: ptr(:)

      ptr => array
      end subroutine

      subroutine F90Array1dAccessMyStruct(ptr,address)
#include <petsc/finclude/petscsys.h>
      use petscsys
      use mymoduleex21f90
      implicit none
      type(MyStruct), pointer :: ptr(:)
      PetscFortranAddr address
      PetscInt start

      start = lbound(ptr,1)
      call F90Array1dGetAddrMyStruct(ptr(start),address)
      end subroutine

      subroutine F90Array1dDestroyMyStruct(ptr)
#include <petsc/finclude/petscsys.h>
      use petscsys
      use mymoduleex21f90
      implicit none
      type(MyStruct), pointer :: ptr(:)

      nullify(ptr)
      end subroutine

      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      use mymoduleex21f90
      implicit none

!
!
!   These two routines are defined in ex21.c they create the Fortran pointer to the derived type
!
      Interface
        Subroutine VecGetArrayMyStruct(v,array,ierr)
          use petscvec
          use mymoduleex21f90
          type(MyStruct), pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
        End Subroutine
      End Interface

      Interface
        Subroutine VecRestoreArrayMyStruct(v,array,ierr)
          use petscvec
          use mymoduleex21f90
          type(MyStruct), pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
        End Subroutine
      End Interface

!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Variables:
!     x, y, w - vectors
!     z       - array of vectors
!
      Vec              x,y
      type(MyStruct),  pointer :: xarray(:)
      PetscInt         n
      PetscErrorCode   ierr
      PetscBool        flg
      integer          i

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'PetscInitialize failed'
        stop
      endif
      n     = 30

      call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr);CHKERRA(ierr)
      call VecCreate(PETSC_COMM_WORLD,x,ierr);CHKERRA(ierr)
      call VecSetSizes(x,PETSC_DECIDE,n,ierr);CHKERRA(ierr)
      call VecSetFromOptions(x,ierr);CHKERRA(ierr)
      call VecDuplicate(x,y,ierr);CHKERRA(ierr)

      call VecGetArrayMyStruct(x,xarray,ierr);CHKERRA(ierr)
      do i=1,10
      xarray(i)%a = i
      xarray(i)%b = 100*i
      xarray(i)%c = 10000*i
      enddo

      call VecRestoreArrayMyStruct(x,xarray,ierr);CHKERRA(ierr)
      call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr);CHKERRA(ierr)
      call VecGetArrayMyStruct(x,xarray,ierr);CHKERRA(ierr)
      do i = 1 , 10
        write(*,*) abs(xarray(i)%a),abs(xarray(i)%b),abs(xarray(i)%c)
      end do
      call VecRestoreArrayMyStruct(x,xarray,ierr);CHKERRA(ierr)

      call VecDestroy(x,ierr);CHKERRA(ierr)
      call VecDestroy(y,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)

      end

!/*TEST
!   build:
!     depends: ex21.c
!
!   test:
!
!TEST*/
