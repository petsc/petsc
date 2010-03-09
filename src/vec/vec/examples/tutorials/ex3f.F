!
!
!  Description: Displays a vector visually.
!
!/*T
!   Concepts: vectors^drawing vectors;
!   Processors: n
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
!  Include petscviewer.h so that we can use the PETSc viewers.
!
#include "finclude/petscsys.h"
#include "finclude/petscvec.h"
#include "finclude/petscviewer.h"

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      Vec     x
      PetscViewer  viewer
      PetscScalar  v
      PetscInt i,istart,iend,n,ione
      PetscErrorCode ierr
      PetscTruth flg

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      n = 50
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER,'-n',n,flg,ierr)

!  Create a vector, specifying only its global dimension.
!  When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
!  the vector format (currently parallel
!  or sequential) is determined at runtime.  Also, the parallel
!  partitioning of the vector is determined by PETSc at runtime.
      call VecCreate(PETSC_COMM_WORLD,x,ierr)
      call VecSetSizes(x,PETSC_DECIDE,n,ierr)
      call VecSetFromOptions(x,ierr)

!  Currently, all PETSc parallel vectors are partitioned by
!  contiguous chunks of rows across the processors.  Determine
!  which vector are locally owned. 
      call VecGetOwnershipRange(x,istart,iend,ierr)

!  Set the vector elements.
!   - Always specify global locations of vector entries.
!   - Each processor needs to insert only elements that it owns locally.
      ione = 1
      do 100 i=istart,iend-1
         v = dble(i)
         call VecSetValues(x,ione,i,v,INSERT_VALUES,ierr)
 100  continue

!  Assemble vector, using the 2-step process:
!    VecAssemblyBegin(), VecAssemblyEnd()
!  Computations can be done while messages are in transition
!  by placing code between these two statements.
      call VecAssemblyBegin(x,ierr)
      call VecAssemblyEnd(x,ierr)

!  Open an X-window viewer.  Note that we specify the same communicator
!  for the viewer as we used for the distributed vector (PETSC_COMM_WORLD).
!    - Helpful runtime option:
!         -draw_pause <pause> : sets time (in seconds) that the
!               program pauses after PetscDrawPause() has been called
!              (0 is default, -1 implies until user input).

      call PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,        &
     &                   PETSC_NULL_CHARACTER,0,0,300,300,viewer,ierr)

!  View the vector
      call VecView(x,viewer,ierr)

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      call PetscViewerDestroy(viewer,ierr)
      call VecDestroy(x,ierr)

      call PetscFinalize(ierr)
      end
 
