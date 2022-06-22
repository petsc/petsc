!
!
!  Description:  Illustrates the use of VecSetValues() to set
!  multiple values at once; demonstrates VecGetArrayF90().
!
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

       PetscCallA(PetscInitialize(ierr))

!  Create initial vector and duplicate it

       PetscCallA(VecCreateSeq(PETSC_COMM_SELF,n,x,ierr))
       PetscCallA(VecDuplicate(x,y,ierr))

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

       PetscCallA(VecSetValues(x,n,loc,xwork,INSERT_VALUES,ierr))

!  Assemble vector

       PetscCallA(VecAssemblyBegin(x,ierr))
       PetscCallA(VecAssemblyEnd(x,ierr))

!  View vector
       PetscCallA(PetscObjectSetName(x, 'initial vector:',ierr))
       PetscCallA(VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr))
       PetscCallA(VecCopy(x,y,ierr))

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArrayF90() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.

       PetscCallA(VecGetArrayF90(x,xx_v,ierr))
       PetscCallA(VecGetArrayF90(y,yy_v,ierr))

!  Modify vector data

       do 30 i=1,n
          xx_v(i) = 100.0*real(i)
          yy_v(i) = 1000.0*real(i)
  30   continue

!  Restore vectors

       PetscCallA(VecRestoreArrayF90(x,xx_v,ierr))
       PetscCallA(VecRestoreArrayF90(y,yy_v,ierr))

!  View vectors
       PetscCallA(PetscObjectSetName(x, 'new vector 1:',ierr))
       PetscCallA(VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr))

       PetscCallA(PetscObjectSetName(y, 'new vector 2:',ierr))
       PetscCallA(VecView(y,PETSC_VIEWER_STDOUT_SELF,ierr))

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

       PetscCallA(VecDestroy(x,ierr))
       PetscCallA(VecDestroy(y,ierr))
       PetscCallA(PetscFinalize(ierr))
       end

!
!/*TEST
!
!     test:
!
!TEST*/
