      program main
      implicit none
!
#include <finclude/petsc.h90>
!
      DM dm
      PetscInt, target, dimension(3) :: EC
      PetscInt, target, dimension(2) :: VE
      PetscInt, pointer :: pEC(:), pVE(:)
      PetscInt, pointer :: nClosure(:)
      PetscInt, pointer :: nJoin(:)
      PetscInt, pointer :: nMeet(:)
      PetscInt       dim, cell, size
      PetscErrorCode ierr

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      CHKERRQ(ierr)
      call DMPlexCreate(PETSC_COMM_WORLD, dm, ierr)
      CHKERRQ(ierr)
      dim = 2
      call DMPlexSetDimension(dm, dim, ierr)
      CHKERRQ(ierr)

! Make Doublet Mesh from Fig 2 of Flexible Representation of Computational Meshes,
! except indexing is from 0 instead of 1 and we obey the new restrictions on
! numbering: cells, vertices, faces, edges
      call DMPlexSetChart(dm, 0, 11, ierr)
      CHKERRQ(ierr)
!     cells
      call DMPlexSetConeSize(dm, 0, 3, ierr)
      CHKERRQ(ierr)
      call DMPlexSetConeSize(dm, 1, 3, ierr)
      CHKERRQ(ierr)
!     edges
      call DMPlexSetConeSize(dm,  6, 2, ierr)
      CHKERRQ(ierr)
      call DMPlexSetConeSize(dm,  7, 2, ierr)
      CHKERRQ(ierr)
      call DMPlexSetConeSize(dm,  8, 2, ierr)
      CHKERRQ(ierr)
      call DMPlexSetConeSize(dm,  9, 2, ierr)
      CHKERRQ(ierr)
      call DMPlexSetConeSize(dm, 10, 2, ierr)
      CHKERRQ(ierr)

      call DMSetUp(dm, ierr)
      CHKERRQ(ierr)

      EC(1) = 6
      EC(2) = 7
      EC(3) = 8
      pEC => EC
      call DMPlexSetCone(dm, 0, pEC, ierr)
      CHKERRQ(ierr)
      EC(1) = 7
      EC(2) = 9
      EC(3) = 10
      pEC => EC
      call DMPlexSetCone(dm, 1 , pEC, ierr)
      CHKERRQ(ierr)

      VE(1) = 2
      VE(2) = 3
      pVE => VE
      call DMPlexSetCone(dm, 6 , pVE, ierr)
      CHKERRQ(ierr)
      VE(1) = 3
      VE(2) = 4
      pVE => VE
      call DMPlexSetCone(dm, 7 , pVE, ierr)
      CHKERRQ(ierr)
      VE(1) = 4
      VE(2) = 2
      pVE => VE
      call DMPlexSetCone(dm, 8 , pVE, ierr)
      CHKERRQ(ierr)
      VE(1) = 3
      VE(2) = 5
      pVE => VE
      call DMPlexSetCone(dm, 9 , pVE, ierr)
      CHKERRQ(ierr)
      VE(1) = 5
      VE(2) = 4
      pVE => VE
      call DMPlexSetCone(dm, 10 , pVE, ierr)
      CHKERRQ(ierr)

      call DMPlexSymmetrize(dm,ierr)
      CHKERRQ(ierr)
      call DMPlexStratify(dm,ierr)
      CHKERRQ(ierr)
      call DMView(dm, PETSC_VIEWER_STDOUT_WORLD, ierr)

!     Test Closure
      do cell = 0,1
         call DMPlexGetTransitiveClosure(dm, cell, PETSC_TRUE,          &
     &        nClosure, ierr)
         CHKERRQ(ierr)
         write(*,*) nClosure
         call DMPlexRestoreTransitiveClosure(dm, cell, PETSC_TRUE,      &
     &        nClosure, ierr)
      end do

!     Test Join
      size  = 2
      VE(1) = 6
      VE(2) = 7
      pVE => VE
      call DMPlexGetJoin(dm, size, pVE, nJoin, ierr)
      write(*,*) 'Join of',pVE,'is',nJoin
      call DMPlexRestoreJoin(dm, size, pVE, nJoin, ierr)
      size  = 2
      VE(1) = 9
      VE(2) = 7
      pVE => VE
      call DMPlexGetJoin(dm, size, pVE, nJoin, ierr)
      write(*,*) 'Join of',pVE,'is',nJoin
      call DMPlexRestoreJoin(dm, size, pVE, nJoin, ierr)
!     Test Full Join
      size  = 3
      EC(1) = 3
      EC(2) = 4
      EC(3) = 5
      pEC => EC
      call DMPlexGetFullJoin(dm, size, pEC, nJoin, ierr)
      write(*,*) 'Full Join of',pEC,'is',nJoin
      call DMPlexRestoreJoin(dm, size, pEC, nJoin, ierr)
!     Test Meet
      size  = 2
      VE(1) = 0
      VE(2) = 1
      pVE => VE
      call DMPlexGetMeet(dm, size, pVE, nMeet, ierr)
      write(*,*) 'Meet of',pVE,'is',nMeet
      call DMPlexRestoreMeet(dm, size, pVE, nMeet, ierr)
      size  = 2
      VE(1) = 6
      VE(2) = 7
      pVE => VE
      call DMPlexGetMeet(dm, size, pVE, nMeet, ierr)
      write(*,*) 'Meet of',pVE,'is',nMeet
      call DMPlexRestoreMeet(dm, size, pVE, nMeet, ierr)

      call DMDestroy(dm, ierr)
      CHKERRQ(ierr)
      call PetscFinalize(ierr)
      CHKERRQ(ierr)
      end
