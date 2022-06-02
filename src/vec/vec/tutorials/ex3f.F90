!
!
!  Description: Displays a vector visually.
!
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      Vec     x
      PetscViewer  viewer
      PetscScalar  v
      PetscInt :: i,istart,iend
      PetscInt, parameter :: ione = 1, n = 50
      PetscErrorCode ierr
      PetscBool  flg

      PetscCallA(PetscInitialize(ierr))
      PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,'-n',n,flg,ierr))

!  Create a vector, specifying only its global dimension.
!  When using VecCreate(), VecSetSizes() and VecSetFromOptions(),
!  the vector format (currently parallel
!  or sequential) is determined at runtime.  Also, the parallel
!  partitioning of the vector is determined by PETSc at runtime.
      PetscCallA(VecCreate(PETSC_COMM_WORLD,x,ierr))
      PetscCallA(VecSetSizes(x,PETSC_DECIDE,n,ierr))
      PetscCallA(VecSetFromOptions(x,ierr))

!  Currently, all PETSc parallel vectors are partitioned by
!  contiguous chunks of rows across the processors.  Determine
!  which vector are locally owned.
      PetscCallA(VecGetOwnershipRange(x,istart,iend,ierr))

!  Set the vector elements.
!   - Always specify global locations of vector entries.
!   - Each processor needs to insert only elements that it owns locally.
      do 100 i=istart,iend-1
         v = 1.0*real(i)
         PetscCallA(VecSetValues(x,ione,i,v,INSERT_VALUES,ierr))
 100  continue

!  Assemble vector, using the 2-step process:
!    VecAssemblyBegin(), VecAssemblyEnd()
!  Computations can be done while messages are in transition
!  by placing code between these two statements.
      PetscCallA(VecAssemblyBegin(x,ierr))
      PetscCallA(VecAssemblyEnd(x,ierr))

!  Open an X-window viewer.  Note that we specify the same communicator
!  for the viewer as we used for the distributed vector (PETSC_COMM_WORLD).
!    - Helpful runtime option:
!         -draw_pause <pause> : sets time (in seconds) that the
!               program pauses after PetscDrawPause() has been called
!              (0 is default, -1 implies until user input).

      PetscCallA(PetscViewerDrawOpen(PETSC_COMM_WORLD,PETSC_NULL_CHARACTER,PETSC_NULL_CHARACTER,0,0,300,300,viewer,ierr))

!  View the vector
      PetscCallA(VecView(x,viewer,ierr))

!  Free work space.  All PETSc objects should be destroyed when they
!  are no longer needed.

      PetscCallA(PetscViewerDestroy(viewer,ierr))
      PetscCallA(VecDestroy(x,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!     test:
!       nsize: 2
!
!TEST*/
