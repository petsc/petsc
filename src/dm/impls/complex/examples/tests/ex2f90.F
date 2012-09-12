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
      call DMComplexCreate(PETSC_COMM_WORLD, dm, ierr)
      CHKERRQ(ierr)
      dim = 2
      call DMComplexSetDimension(dm, dim, ierr)
      CHKERRQ(ierr)

! Make Doublet Mesh from Fig 2 of Flexible Representation of Computational Meshes,
! except indexing is from 0 instead of 1 and we obey the new restrictions on
! numbering: cells, vertices, faces, edges
      call DMComplexSetChart(dm, 0, 11, ierr)
      CHKERRQ(ierr)
!     cells
      call DMComplexSetConeSize(dm, 0, 3, ierr)
      CHKERRQ(ierr)
      call DMComplexSetConeSize(dm, 1, 3, ierr)
      CHKERRQ(ierr)
!     edges
      call DMComplexSetConeSize(dm,  6, 2, ierr)
      CHKERRQ(ierr)
      call DMComplexSetConeSize(dm,  7, 2, ierr)
      CHKERRQ(ierr)
      call DMComplexSetConeSize(dm,  8, 2, ierr)
      CHKERRQ(ierr)
      call DMComplexSetConeSize(dm,  9, 2, ierr)
      CHKERRQ(ierr)
      call DMComplexSetConeSize(dm, 10, 2, ierr)
      CHKERRQ(ierr)

      call DMSetUp(dm, ierr)
      CHKERRQ(ierr)

      EC(1) = 6
      EC(2) = 7
      EC(3) = 8
      pEC => EC
      call DMComplexSetCone(dm, 0, pEC, ierr)
      CHKERRQ(ierr)
      EC(1) = 7
      EC(2) = 9
      EC(3) = 10
      pEC => EC
      call DMComplexSetCone(dm, 1 , pEC, ierr)
      CHKERRQ(ierr)

      VE(1) = 2
      VE(2) = 3
      pVE => VE
      call DMComplexSetCone(dm, 6 , pVE, ierr)
      CHKERRQ(ierr)
      VE(1) = 3
      VE(2) = 4
      pVE => VE
      call DMComplexSetCone(dm, 7 , pVE, ierr)
      CHKERRQ(ierr)
      VE(1) = 4
      VE(2) = 2
      pVE => VE
      call DMComplexSetCone(dm, 8 , pVE, ierr)
      CHKERRQ(ierr)
      VE(1) = 3
      VE(2) = 5
      pVE => VE
      call DMComplexSetCone(dm, 9 , pVE, ierr)
      CHKERRQ(ierr)
      VE(1) = 5
      VE(2) = 4
      pVE => VE
      call DMComplexSetCone(dm, 10 , pVE, ierr)
      CHKERRQ(ierr)

      call DMComplexSymmetrize(dm,ierr)
      CHKERRQ(ierr)
      call DMComplexStratify(dm,ierr)
      CHKERRQ(ierr)
      call DMView(dm, PETSC_VIEWER_STDOUT_WORLD, ierr)

!     Test Closure
      do cell = 0,1
         call DMComplexGetTransitiveClosure(dm, cell, PETSC_TRUE,       &
     &        nClosure, ierr)
         CHKERRQ(ierr)
         write(*,*) nClosure
         call DMComplexRestoreTransitiveClosure(dm, cell, PETSC_TRUE,   &
     &        nClosure, ierr)
      end do

!     Test Join
      size  = 2
      VE(1) = 6
      VE(2) = 7
      pVE => VE
      call DMComplexGetJoin(dm, size, pVE, nJoin, ierr)
      write(*,*) 'Join of',pVE,'is',nJoin
      call DMComplexRestoreJoin(dm, size, pVE, nJoin, ierr)
      size  = 2
      VE(1) = 9
      VE(2) = 7
      pVE => VE
      call DMComplexGetJoin(dm, size, pVE, nJoin, ierr)
      write(*,*) 'Join of',pVE,'is',nJoin
      call DMComplexRestoreJoin(dm, size, pVE, nJoin, ierr)
!     Test Full Join
      size  = 3
      EC(1) = 3
      EC(2) = 4
      EC(3) = 5
      pEC => EC
      call DMComplexGetFullJoin(dm, size, pEC, nJoin, ierr)
      write(*,*) 'Full Join of',pEC,'is',nJoin
      call DMComplexRestoreJoin(dm, size, pEC, nJoin, ierr)
!     Test Meet
      size  = 2
      VE(1) = 0
      VE(2) = 1
      pVE => VE
      call DMComplexGetMeet(dm, size, pVE, nMeet, ierr)
      write(*,*) 'Meet of',pVE,'is',nMeet
      call DMComplexRestoreMeet(dm, size, pVE, nMeet, ierr)
      size  = 2
      VE(1) = 6
      VE(2) = 7
      pVE => VE
      call DMComplexGetMeet(dm, size, pVE, nMeet, ierr)
      write(*,*) 'Meet of',pVE,'is',nMeet
      call DMComplexRestoreMeet(dm, size, pVE, nMeet, ierr)

      call DMDestroy(dm, ierr)
      CHKERRQ(ierr)
      call PetscFinalize(ierr)
      CHKERRQ(ierr)
      end
