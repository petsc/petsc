!
!
!  Description:  Illustrates the use of VecSetValues() to set
!  multiple values at once; demonstrates VecGetArray().
!
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Macro definitions
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  Macros to make clearer the process of setting values in vectors and
!  getting values from vectors.
!
!   - The element xx_a(ib) is element ib+1 in the vector x
!   - Here we add 1 to the base array index to facilitate the use of
!     conventional Fortran 1-based array indexing.
!
#define xx_a(ib)  xx_v(xx_i + (ib))
#define yy_a(ib)  yy_v(yy_i + (ib))

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

       PetscScalar xwork(6)
       PetscScalar xx_v(1),yy_v(1)
       PetscInt     i,n,loc(6),isix
       PetscErrorCode ierr
       PetscOffset xx_i,yy_i
       Vec         x,y

       PetscCallA(PetscInitialize(ierr))
       n = 6
       isix = 6

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

       PetscCallA(VecSetValues(x,isix,loc,xwork,INSERT_VALUES,ierr))

!  Assemble vector

       PetscCallA(VecAssemblyBegin(x,ierr))
       PetscCallA(VecAssemblyEnd(x,ierr))

!  View vector
       PetscCallA(PetscObjectSetName(x, 'initial vector:',ierr))
       PetscCallA(VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr))
       PetscCallA(VecCopy(x,y,ierr))

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the users manual for details.

       PetscCallA(VecGetArray(x,xx_v,xx_i,ierr))
       PetscCallA(VecGetArray(y,yy_v,yy_i,ierr))

!  Modify vector data

       do 30 i=1,n
          xx_a(i) = 100.0*real(i)
          yy_a(i) = 1000.0*real(i)
  30   continue

!  Restore vectors

       PetscCallA(VecRestoreArray(x,xx_v,xx_i,ierr))
       PetscCallA(VecRestoreArray(y,yy_v,yy_i,ierr))

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

!/*TEST
!
!     test:
!
!TEST*/
