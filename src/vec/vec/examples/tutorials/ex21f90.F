!
!
!    Demonstrates how one may access entries of a PETSc Vec as if it was an array of Fortran derived types
!
!
! -----------------------------------------------------------------------
#include <petsc/finclude/petscsysdef.h>
#include <petsc/finclude/petscvecdef.h>

      module mymodule
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
      use mymodule
      implicit none
#include <petsc/finclude/petscsys.h>
      PetscInt start,len
      type(MyStruct), target ::                                               &
     &             array(start:start+len-1)
      type(MyStruct), pointer :: ptr(:)

      ptr => array
      end subroutine

      subroutine F90Array1dAccessMyStruct(ptr,address)
      use mymodule
      implicit none
#include <petsc/finclude/petscsys.h>
      type(MyStruct), pointer :: ptr(:)
      PetscFortranAddr address
      PetscInt start

      start = lbound(ptr,1)
      call F90Array1dGetAddrMyStruct(ptr(start),address)
      end subroutine

      subroutine F90Array1dDestroyMyStruct(ptr)
      use mymodule
      implicit none
#include <petsc/finclude/petscsys.h>
      type(MyStruct), pointer :: ptr(:)

      nullify(ptr)
      end subroutine


      program main
      use mymodule
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  The following include statements are required for Fortran programs
!  that use PETSc vectors:
!     petscsys.h       - base PETSc routines
!     petscvec.h    - vectors
!     petscvec.h90  - to allow access to Fortran90 features of vectors
!
!  Additional include statements may be needed if using additional
!  PETSc routines in a Fortran program, e.g.,
!     petscviewer.h - viewers
!     petscis.h     - index sets
!
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscviewer.h>
#include <petsc/finclude/petscvec.h>
#include <petsc/finclude/petscvec.h90>

!
!   These two routines are defined in ex21.c they create the Fortran pointer to the derived type
!
      Interface
        Subroutine VecGetArrayMyStruct(v,array,ierr)
          use mymodule
          type(MyStruct), pointer :: array(:)
          PetscErrorCode ierr
          Vec     v
        End Subroutine
      End Interface

      Interface
        Subroutine VecRestoreArrayMyStruct(v,array,ierr)
          use mymodule
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
      n     = 30

      call PetscOptionsGetInt(PETSC_NULL_OBJECT,PETSC_NULL_CHARACTER,    &
     &                        '-n',n,flg,ierr)
      call VecCreate(PETSC_COMM_WORLD,x,ierr)
      call VecSetSizes(x,PETSC_DECIDE,n,ierr)
      call VecSetFromOptions(x,ierr)
      call VecDuplicate(x,y,ierr)

      call VecGetArrayMyStruct(x,xarray,ierr)
      do i=1,10
      xarray(i)%a = i
      xarray(i)%b = 100*i
      xarray(i)%c = 10000*i
      enddo

      call VecRestoreArrayMyStruct(x,xarray,ierr)
      call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr)
      call VecGetArrayMyStruct(x,xarray,ierr)
      do i = 1 , 10
        write(*,*) abs(xarray(i)%a),abs(xarray(i)%b),abs(xarray(i)%c)
      end do
      call VecRestoreArrayMyStruct(x,xarray,ierr)


      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)
      call PetscFinalize(ierr)

      end

