
C      "$Id: ex4f90.F,v 1.2 1997/01/22 18:41:37 bsmith Exp bsmith $";

C  Description:  Illustrates the use of VecSetValues() to set
C  multiple values at once; demonstrates VecGetArrayF90().
C
C/*T
C   Concepts: Vectors^Assembling vectors; Using vector arrays;
C   Routines: VecCreateSeq(); VecDuplicate(); VecSetValues(); VecView();
C   Routines: VecCopy(); VecView(); VecGetArrayF90(); VecRestoreArrayF90();
C   Routines: VecAssemblyBegin(); VecAssemblyEnd(); VecDestroy();
C   Processors: 1
CT*/
C -----------------------------------------------------------------------

      program ex4f90
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  The following include statements are required for Fortran programs
C  that use PETSc vectors:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors
C     vec.h90  - to allow access to Fortran 90 features of PETSc

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/vec.h90"

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

       Scalar  xwork(6)
       Scalar, pointer ::  xx_v(:), yy_v(:)
       integer i, n, ierr, loc(6)
       Vec     x, y


       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
       n = 6

C  Create initial vector and duplicate it

       call VecCreateSeq(MPI_COMM_SELF,n,x,ierr)
       call VecDuplicate(x,y,ierr)

C  Fill work arrays with vector entries and locations.  Note that
C  the vector indices are 0-based in PETSc (for both Fortran and
C  C vectors)

       do 10 i=1,n
          loc(i) = i-1
          xwork(i) = 10.0*i
  10   continue

C  Set vector values.  Note that we set multiple entries at once.
C  Of course, usually one would create a work array that is the
C  natural size for a particular problem (not one that is as long
C  as the full vector).

       call VecSetValues(x,6,loc,xwork,INSERT_VALUES,ierr)

C  Assemble vector

       call VecAssemblyBegin(x,ierr)
       call VecAssemblyEnd(x,ierr)

C  View vector

       write(6,20)
  20   format('initial vector:')
       call VecView(x,VIEWER_STDOUT_SELF,ierr)
       call VecCopy(x,y,ierr)

C  Get a pointer to vector data.
C    - For default PETSc vectors, VecGetArrayF90() returns a pointer to
C      the data array.  Otherwise, the routine is implementation dependent.
C    - You MUST call VecRestoreArray() when you no longer need access to
C      the array.

       call VecGetArrayF90(x,xx_v,ierr)
       call VecGetArrayF90(y,yy_v,ierr)

C  Modify vector data

       do 30 i=1,n
          xx_v(i) = 100.0*i
          yy_v(i) = 1000.0*i
  30   continue

C  Restore vectors

       call VecRestoreArrayF90(x,xx_v,ierr)
       call VecRestoreArrayF90(y,yy_v,ierr)

C  View vectors

       write(6,40)
  40   format('new vector 1:')
       call VecView(x,VIEWER_STDOUT_SELF,ierr)

       write(6,50)
  50   format('new vector 2:')
       call VecView(y,VIEWER_STDOUT_SELF,ierr)

C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.

       call VecDestroy(x,ierr)
       call VecDestroy(y,ierr)
       call PetscFinalize(ierr)
       end
 
