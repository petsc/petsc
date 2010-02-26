!
!
!  Description:  Illustrates the use of VecSetValues() to set
!  multiple values at once; demonstrates VecGetArray().
!
!/*T
!   Concepts: vectors^assembling;
!   Concepts: vectors^arrays of vectors;
!   Processors: 1
!T*/
! -----------------------------------------------------------------------

      program main
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  The following include statements are required for Fortran programs
!  that use PETSc vectors:
!     petscsys.h       - base PETSc routines
!     petscvec.h    - vectors

#include "finclude/petscsys.h"
#include "finclude/petscvec.h"

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

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
       n = 6
       isix = 6

!  Create initial vector and duplicate it

       call VecCreateSeq(PETSC_COMM_SELF,n,x,ierr)
       call VecDuplicate(x,y,ierr)

!  Fill work arrays with vector entries and locations.  Note that
!  the vector indices are 0-based in PETSc (for both Fortran and
!  C vectors)

       do 10 i=1,n
          loc(i) = i-1
          xwork(i) = 10.0*i
  10   continue

!  Set vector values.  Note that we set multiple entries at once.
!  Of course, usually one would create a work array that is the
!  natural size for a particular problem (not one that is as long
!  as the full vector).

       call VecSetValues(x,isix,loc,xwork,INSERT_VALUES,ierr)

!  Assemble vector

       call VecAssemblyBegin(x,ierr)
       call VecAssemblyEnd(x,ierr)

!  View vector

       write(6,20)
  20   format('initial vector:')
       call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr)
       call VecCopy(x,y,ierr)

!  Get a pointer to vector data.
!    - For default PETSc vectors, VecGetArray() returns a pointer to
!      the data array.  Otherwise, the routine is implementation dependent.
!    - You MUST call VecRestoreArray() when you no longer need access to
!      the array.
!    - Note that the Fortran interface to VecGetArray() differs from the
!      C version.  See the users manual for details.

       call VecGetArray(x,xx_v,xx_i,ierr)
       call VecGetArray(y,yy_v,yy_i,ierr)

!  Modify vector data

       do 30 i=1,n
          xx_a(i) = 100.0*i
          yy_a(i) = 1000.0*i
  30   continue

!  Restore vectors

       call VecRestoreArray(x,xx_v,xx_i,ierr)
       call VecRestoreArray(y,yy_v,yy_i,ierr)

!  View vectors

       write(6,40)
  40   format('new vector 1:')
       call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr)

       write(6,50)
  50   format('new vector 2:')
       call VecView(y,PETSC_VIEWER_STDOUT_SELF,ierr)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

       call VecDestroy(x,ierr)
       call VecDestroy(y,ierr)
       call PetscFinalize(ierr)
       end
 
