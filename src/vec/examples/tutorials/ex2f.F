
C    "$Id: ex2f.F,v 1.1 1996/10/26 18:34:22 curfman Exp curfman $"

C  Description: Builds a parallel vector with 1 component on the first
C               processor, 2 on the second, etc.  Then each processor adds
C               one to all elements except the last rank.
C
C/*T
C   Concepts: Vectors^Assembling vectors;
C   Routines: VecCreateMPI(); VecGetSize(); VecSet(); VecSetValues();
C   Routines: VecView(); VecDestroy();
C   Processors: n
CT*/
C -----------------------------------------------------------------------

      program ex2f
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  The following include statements are required for Fortran programs
C  that use PETSc vectors:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors
C  Additional include statements may be needed if using additional
C  PETSc routines in a Fortran program, e.g.,
C     viewer.h - viewers
C     is.h     - index sets
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
C
      Vec     x
      integer N, ierr, rank, i
      Scalar  one

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      one   = 1.0
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

C  Create a parallel vector.
C   - In this case, we specify the size of each processor's local
C     portion, and PETSc computes the global size.  Alternatively,
C     PETSc could determine the vector's distribution if we specify
C     just the global size.

      call VecCreateMPI(MPI_COMM_WORLD,rank+1,PETSC_DECIDE,x,ierr)
      call VecGetSize(x,N,ierr)
      call VecSet(one,x,ierr)

C  Set the vector elements.
C   - Note that VecSetValues() uses 0-based row and column numbers
C     in Fortran as well as in C.
C   - Always specify global locations of vector entries.
C   - Each processor can contribute any vector entries,
C     regardless of which processor "owns" them; any nonlocal
C     contributions will be transferred to the appropriate processor
C     during the assembly process.
C   - In this example, the flag ADD_VALUES indicates that all
C     contributions will be added together.

      do 100 i=0, N-rank-1
         call VecSetValues(x,1,i,one,ADD_VALUES,ierr)
 100  continue

C  Assemble vector, using the 2-step process:
C    VecAssemblyBegin(), VecAssemblyEnd()
C  Computations can be done while messages are in transition
C  by placing code between these two statements.

      call VecAssemblyBegin(x,ierr)
      call VecAssemblyEnd(x,ierr)

C  View the vector; then destroy it.

      call VecView(x,VIEWER_STDOUT_WORLD,ierr)
      call VecDestroy(x,ierr)

      call PetscFinalize(ierr)
      end
 
