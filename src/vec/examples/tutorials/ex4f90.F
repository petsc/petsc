
C      "$Id: ex4f.F,v 1.16 1996/11/27 22:51:13 bsmith Exp $";

C  Description:  Illustrates the use of VecSetValues() to set
C  multiple values at once; demonstrates VecGetArray().
C
C/*T
C   Concepts: Vectors^Assembling vectors; Using vector arrays;
C   Routines: VecCreateSeq(); VecDuplicate(); VecSetValues(); VecView();
C   Routines: VecCopy(); VecView(); VecGetArray(); VecRestoreArray();
C   Routines: VecAssemblyBegin(); VecAssemblyEnd(); VecDestroy();
C   Processors: 1
CT*/
C -----------------------------------------------------------------------

      program ex4f
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  The following include statements are required for Fortran programs
C  that use PETSc vectors:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C                   Macro definitions
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
C
C  Macros to make clearer the process of setting values in vectors and
C  getting values from vectors.
C
C   - The element xx_a(ib) is element ib+1 in the vector x
C   - Here we add 1 to the base array index to facilitate the use of
C     conventional Fortran 1-based array indexing.
C
#define xx_a(ib)  xx_v(xx_i + (ib))
#define yy_a(ib)  yy_v(yy_i + (ib))

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

       Scalar  xwork(6)
       Scalar  xx_v(1), yy_v(1)
       integer i, n, ierr, loc(6), xx_i, yy_i
       Vec     x, y
       common xx_v, yy_v

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
C    - For default PETSc vectors, VecGetArray() returns a pointer to
C      the data array.  Otherwise, the routine is implementation dependent.
C    - You MUST call VecRestoreArray() when you no longer need access to
C      the array.
C    - Note that the Fortran interface to VecGetArray() differs from the
C      C version.  See the users manual for details.

       call VecGetArray(x,xx_v,xx_i,ierr)
       call VecGetArray(y,yy_v,yy_i,ierr)

C  Modify vector data

       do 30 i=1,n
          xx_a(i) = 100.0*i
          yy_a(i) = 1000.0*i
  30   continue

C  Restore vectors

       call VecRestoreArray(x,xx_v,xx_i,ierr)
       call VecRestoreArray(y,yy_v,yy_i,ierr)

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
 
