      program main
      implicit none
!
#include <petsc/finclude/petsc.h90>
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
      call DMPlexCreate(PETSC_COMM_WORLD, dm, ierr)
      firstCell = 0
      numCells = 2
      numVertices = 6
      numPoints = numCells+numVertices
      call DMPlexSetChart(dm, i0, numPoints, ierr)
      do c=firstCell,numCells-1
         call DMPlexSetConeSize(dm, c, i4, ierr)
      end do
      call DMSetUp(dm, ierr)

      EC(1) = 2
      EC(2) = 3
      EC(3) = 4
      EC(4) = 5
      pEC => EC
      c = 0
      write(*,*) 'cell',c,pEC
      call DMPlexSetCone(dm, c , pEC, ierr)
      CHKERRQ(ierr)
      call DMPlexGetCone(dm, c , pEC, ierr)
      CHKERRQ(ierr)
      write(*,*) 'cell',c,pEC
      EC(1) = 4
      EC(2) = 5
      EC(3) = 6
      EC(4) = 7
      pEC => EC
      c = 1
      write(*,*) 'cell',c,pEC
      call DMPlexSetCone(dm, c , pEC, ierr)
      CHKERRQ(ierr)
      call DMPlexGetCone(dm, c , pEC, ierr)
      CHKERRQ(ierr)
      write(*,*) 'cell',c,pEC
      call DMPlexRestoreCone(dm, c , pEC, ierr)
      CHKERRQ(ierr)

      call DMPlexSymmetrize(dm, ierr)
      call DMPlexStratify(dm, ierr)

      v = 4
      call DMPlexGetSupport(dm, v , pES, ierr)
      CHKERRQ(ierr)
      write(*,*) 'vertex',v,pES
      call DMPlexRestoreSupport(dm, v , pES, ierr)
      CHKERRQ(ierr)

      call DMDestroy(dm,ierr)
      call PetscFinalize(ierr)
      end
