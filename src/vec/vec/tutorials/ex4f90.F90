!
!
!  Description:  Illustrates the use of VecSetValues() to set
!  multiple values at once; demonstrates VecGetArrayF90().
!
!/*T
!   Concepts: vectors^assembling vectors;
!   Concepts: vectors^arrays;
!   Concepts: Fortran90^assembling vectors;
!   Processors: 1
!T*/
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

       PetscInt, parameter :: n=6
       PetscScalar, dimension(n) ::  xwork
       PetscScalar, pointer, dimension(:) ::  xx_v,yy_v
       PetscInt, dimension(n) :: loc
       PetscInt i
       PetscErrorCode ierr
       Vec     x,y

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
       if (ierr /= 0) then
         print*,'PetscInitialize failed'
         stop
       endif

!  Create initial vector and duplicate it

       call VecCreateSeq(PETSC_COMM_SELF,n,x,ierr);CHKERRA(ierr)
       call VecDuplicate(x,y,ierr);CHKERRA(ierr)

!  Fill work arrays with vector entries and locations.  Note that
!  the vector indices are 0-based in PETSc (for both Fortran and
!  C vectors)

       do 10 i=1,n
          loc(i) = i-1
          xwork(i) = 10.0*real(i)
  10   continue

!  Set vector values.  Note that we set multiple entries at once.
!  Of course, usually one would create a work array that is the
!  natural size for a particular problem (not one that is as long
!  as the full vector).

       call VecSetValues(x,n,loc,xwork,INSERT_VALUES,ierr);CHKERRA(ierr)

!  Assemble vector

       call VecAssemblyBegin(x,ierr);CHKERRA(ierr)
       call VecAssemblyEnd(x,ierr);CHKERRA(ierr)

!  View vector
       call PetscObjectSetName(x, 'initial vector:',ierr);CHKERRA(ierr)
       call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr);CHKERRA(ierr)
       call VecCopy(x,y,ierr);CHKERRA(ierr)

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArrayF90() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.

       call VecGetArrayF90(x,xx_v,ierr);CHKERRA(ierr)
       call VecGetArrayF90(y,yy_v,ierr);CHKERRA(ierr)

!  Modify vector data

       do 30 i=1,n
          xx_v(i) = 100.0*real(i)
          yy_v(i) = 1000.0*real(i)
  30   continue

!  Restore vectors

       call VecRestoreArrayF90(x,xx_v,ierr);CHKERRA(ierr)
       call VecRestoreArrayF90(y,yy_v,ierr);CHKERRA(ierr)

!  View vectors
       call PetscObjectSetName(x, 'new vector 1:',ierr);CHKERRA(ierr)
       call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr);CHKERRA(ierr)

       call PetscObjectSetName(y, 'new vector 2:',ierr);CHKERRA(ierr)
       call VecView(y,PETSC_VIEWER_STDOUT_SELF,ierr);CHKERRA(ierr)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

       call VecDestroy(x,ierr);CHKERRA(ierr)
       call VecDestroy(y,ierr);CHKERRA(ierr)
       call PetscFinalize(ierr)
       end

!
!/*TEST
!
!     test:
!
!TEST*/
