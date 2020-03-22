      program main
#include <petsc/finclude/petscdmplex.h>
      use petscdmplex
      use petscsys
      implicit none

      DM dm
      PetscInt, target, dimension(3) :: EC
      PetscInt, target, dimension(2) :: VE
      PetscInt, pointer :: pEC(:), pVE(:)
      PetscInt, pointer :: nClosure(:)
      PetscInt, pointer :: nJoin(:)
      PetscInt, pointer :: nMeet(:)
      PetscInt       dim, cell, size
      PetscInt i0,i1,i2,i3,i4,i5,i6,i7
      PetscInt i8,i9,i10,i11
      PetscErrorCode ierr

      i0 = 0
      i1 = 1
      i2 = 2
      i3 = 3
      i4 = 4
      i5 = 5
      i6 = 6
      i7 = 7
      i8 = 8
      i9 = 9
      i10 = 10
      i11 = 11

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      call DMPlexCreate(PETSC_COMM_WORLD, dm, ierr);CHKERRA(ierr)
      call PetscObjectSetName(dm, 'Mesh', ierr);CHKERRA(ierr)
      dim = 2
      call DMSetDimension(dm, dim, ierr);CHKERRA(ierr)

! Make Doublet Mesh from Fig 2 of Flexible Representation of Computational Meshes,
! except indexing is from 0 instead of 1 and we obey the new restrictions on
! numbering: cells, vertices, faces, edges
      call DMPlexSetChart(dm, i0, i11, ierr);CHKERRA(ierr)
!     cells
      call DMPlexSetConeSize(dm, i0, i3, ierr);CHKERRA(ierr)
      call DMPlexSetConeSize(dm, i1, i3, ierr);CHKERRA(ierr)
!     edges
      call DMPlexSetConeSize(dm,  i6, i2, ierr);CHKERRA(ierr)
      call DMPlexSetConeSize(dm,  i7, i2, ierr);CHKERRA(ierr)
      call DMPlexSetConeSize(dm,  i8, i2, ierr);CHKERRA(ierr)
      call DMPlexSetConeSize(dm,  i9, i2, ierr);CHKERRA(ierr)
      call DMPlexSetConeSize(dm, i10, i2, ierr);CHKERRA(ierr)

      call DMSetUp(dm, ierr);CHKERRA(ierr)

      EC(1) = 6
      EC(2) = 7
      EC(3) = 8
      pEC => EC
      call DMPlexSetCone(dm, i0, pEC, ierr);CHKERRA(ierr)
      EC(1) = 7
      EC(2) = 9
      EC(3) = 10
      pEC => EC
      call DMPlexSetCone(dm, i1 , pEC, ierr);CHKERRA(ierr)

      VE(1) = 2
      VE(2) = 3
      pVE => VE
      call DMPlexSetCone(dm, i6 , pVE, ierr);CHKERRA(ierr)
      VE(1) = 3
      VE(2) = 4
      pVE => VE
      call DMPlexSetCone(dm, i7 , pVE, ierr);CHKERRA(ierr)
      VE(1) = 4
      VE(2) = 2
      pVE => VE
      call DMPlexSetCone(dm, i8 , pVE, ierr);CHKERRA(ierr)
      VE(1) = 3
      VE(2) = 5
      pVE => VE
      call DMPlexSetCone(dm, i9 , pVE, ierr);CHKERRA(ierr)
      VE(1) = 5
      VE(2) = 4
      pVE => VE
      call DMPlexSetCone(dm, i10 , pVE, ierr);CHKERRA(ierr)

      call DMPlexSymmetrize(dm,ierr);CHKERRA(ierr)
      call DMPlexStratify(dm,ierr);CHKERRA(ierr)
      call DMView(dm, PETSC_VIEWER_STDOUT_WORLD, ierr);CHKERRA(ierr)

!     Test Closure
      do cell = 0,1
         call DMPlexGetTransitiveClosure(dm,cell,PETSC_TRUE,nClosure,ierr);CHKERRA(ierr)
!     Different Fortran compilers print a different number of columns
!     per row producing different outputs in the test runs hence
!     do not print the nClosure
        write(*,1000) 'nClosure ',nClosure
 1000   format (a,30i4)
        call DMPlexRestoreTransitiveClosure(dm,cell,PETSC_TRUE,nClosure,ierr);CHKERRA(ierr)
      end do

!     Test Join
      size  = 2
      VE(1) = 6
      VE(2) = 7
      pVE => VE
      call DMPlexGetJoin(dm, size, pVE, nJoin, ierr);CHKERRA(ierr)
      write(*,1001) 'Join of',pVE
      write(*,1002) '  is',nJoin
      call DMPlexRestoreJoin(dm, size, pVE, nJoin, ierr);CHKERRA(ierr)
      size  = 2
      VE(1) = 9
      VE(2) = 7
      pVE => VE
      call DMPlexGetJoin(dm, size, pVE, nJoin, ierr);CHKERRA(ierr)
      write(*,1001) 'Join of',pVE
 1001 format (a,10i5)
       write(*,1002) '  is',nJoin
 1002  format (a,10i5)
     call DMPlexRestoreJoin(dm, size, pVE, nJoin, ierr);CHKERRA(ierr)
!     Test Full Join
      size  = 3
      EC(1) = 3
      EC(2) = 4
      EC(3) = 5
      pEC => EC
      call DMPlexGetFullJoin(dm, size, pEC, nJoin, ierr);CHKERRA(ierr)
      write(*,1001) 'Full Join of',pEC
      write(*,1002) '  is',nJoin
      call DMPlexRestoreJoin(dm, size, pEC, nJoin, ierr);CHKERRA(ierr)
!     Test Meet
      size  = 2
      VE(1) = 0
      VE(2) = 1
      pVE => VE
      call DMPlexGetMeet(dm, size, pVE, nMeet, ierr);CHKERRA(ierr)
      write(*,1001) 'Meet of',pVE
      write(*,1002) '  is',nMeet
      call DMPlexRestoreMeet(dm, size, pVE, nMeet, ierr);CHKERRA(ierr)
      size  = 2
      VE(1) = 6
      VE(2) = 7
      pVE => VE
      call DMPlexGetMeet(dm, size, pVE, nMeet, ierr);CHKERRA(ierr)
      write(*,1001) 'Meet of',pVE
      write(*,1002) '  is',nMeet
      call DMPlexRestoreMeet(dm, size, pVE, nMeet, ierr);CHKERRA(ierr)

      call DMDestroy(dm, ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end
!
!/*TEST
!
!   test:
!     suffix: 0
!
!TEST*/
