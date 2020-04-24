      program main
#include <petsc/finclude/petscdmplex.h>
      use petscdmplex
      use petscsys
      implicit none
!
!
      DM dm
      PetscInt, target, dimension(4) :: EC
      PetscInt, pointer :: pEC(:)
      PetscInt, pointer :: pES(:)
      PetscInt c, firstCell, numCells
      PetscInt v, numVertices, numPoints
      PetscInt i0,i4
      PetscErrorCode ierr

      i0 = 0
      i4 = 4

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
     if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call DMPlexCreate(PETSC_COMM_WORLD, dm, ierr);CHKERRA(ierr)
      firstCell = 0
      numCells = 2
      numVertices = 6
      numPoints = numCells+numVertices
      call DMPlexSetChart(dm, i0, numPoints, ierr);CHKERRA(ierr)
      do c=firstCell,numCells-1
         call DMPlexSetConeSize(dm, c, i4, ierr);CHKERRA(ierr)
      end do
      call DMSetUp(dm, ierr);CHKERRA(ierr)

      EC(1) = 2
      EC(2) = 3
      EC(3) = 4
      EC(4) = 5
      pEC => EC
      c = 0
      write(*,1000) 'cell',c,pEC
 1000 format (a,i4,50i4)
      call DMPlexSetCone(dm, c , pEC, ierr);CHKERRA(ierr)
      call DMPlexGetCone(dm, c , pEC, ierr);CHKERRA(ierr)
      write(*,1000) 'cell',c,pEC
      EC(1) = 4
      EC(2) = 5
      EC(3) = 6
      EC(4) = 7
      pEC => EC
      c = 1
      write(*,1000) 'cell',c,pEC
      call DMPlexSetCone(dm, c , pEC, ierr);CHKERRA(ierr)
      call DMPlexGetCone(dm, c , pEC, ierr);CHKERRA(ierr)
      write(*,1000) 'cell',c,pEC
      call DMPlexRestoreCone(dm, c , pEC, ierr);CHKERRA(ierr)

      call DMPlexSymmetrize(dm, ierr);CHKERRA(ierr)
      call DMPlexStratify(dm, ierr);CHKERRA(ierr)

      v = 4
      call DMPlexGetSupport(dm, v , pES, ierr);CHKERRA(ierr)
      write(*,1000) 'vertex',v,pES
      call DMPlexRestoreSupport(dm, v , pES, ierr);CHKERRA(ierr)

      call DMDestroy(dm,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

! /*TEST
!
! test:
!   suffix: 0
!
! TEST*/
