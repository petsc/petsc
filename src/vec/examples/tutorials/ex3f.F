
C  Description: Displays a vector visually.
C
C/*T
C   Concepts: Vectors^Drawing vectors;
C   Routines: VecCreate(); VecSetValues(); VecView(); VecDestroy();
C   Routines: ViewerDrawOpenX(); ViewerDestroy();
C   Processors: n
CT*/
C -----------------------------------------------------------------------

      program ex3f
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  The following include statements are required for Fortran programs
C  that use PETSc vectors:
C     petsc.h  - base PETSc routines
C     vec.h    - vectors
C  Include viewer.h so that we can use the PETSc viewers.
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/viewer.h"
C
      Vec     x
      Viewer  viewer
      Scalar  v
      integer i, istart, iend, n, ierr, flg

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      n = 50
      call OptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

C  Create a vector, specifying only its global dimension.
C  When using VecCreate(), the vector format (currently parallel
C  or sequential) is determined at runtime.  Also, the parallel
C  partitioning of the vector is determined by PETSc at runtime.
      call VecCreate(MPI_COMM_WORLD,n,x,ierr)

C  Currently, all PETSc parallel vectors are partitioned by
C  contiguous chunks of rows across the processors.  Determine
C  which vector are locally owned. 
      call VecGetOwnershipRange(x,istart,iend,ierr)

C  Set the vector elements.
C   - Always specify global locations of vector entries.
C   - Each processor needs to insert only elements that it owns locally.
      do 100 i=istart,iend-1
         v = dble(i)
         call VecSetValues(x,1,i,v,INSERT_VALUES,ierr)
 100  continue

C  Assemble vector, using the 2-step process:
C    VecAssemblyBegin(), VecAssemblyEnd()
C  Computations can be done while messages are in transition
C  by placing code between these two statements.
      call VecAssemblyBegin(x,ierr)
      call VecAssemblyEnd(x,ierr)

C  Open an X-window viewer.  Note that we specify the same communicator
C  for the viewer as we used for the distributed vector (MPI_COMM_WORLD).
C    - Helpful runtime option:
C         -draw_pause <pause> : sets time (in seconds) that the
C               program pauses after DrawPause() has been called
C              (0 is default, -1 implies until user input).

      call ViewerDrawOpenX(MPI_COMM_WORLD,PETSC_NULL_CHARACTER,
     &                     PETSC_NULL_CHARACTER,0,0,300,300,viewer,ierr)

C  View the vector
      call VecView(x,viewer,ierr)

C  Free work space.  All PETSc objects should be destroyed when they
C  are no longer needed.

      call ViewerDestroy(viewer,ierr)
      call VecDestroy(x,ierr)

      call PetscFinalize(ierr)
      end
 
