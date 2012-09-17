      program main
      implicit none
!
#include <finclude/petsc.h90>
!
      DM dm
      PetscInt, target, dimension(4) :: EC
      PetscInt, pointer :: pEC(:)
      PetscInt c,d, firstCell, numCells
      PetscErrorCode ierr

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call DMComplexCreate(PETSC_COMM_WORLD, dm, ierr)
      firstCell = 0
      numCells = 5
      call DMComplexSetChart(dm, 0, numCells, ierr)
      do c=firstCell,numCells-1
         call DMComplexSetConeSize(dm, c, 4, ierr)
      end do
      call DMSetUp(dm, ierr)

      EC(1) = 1
      EC(2) = 2
      EC(3) = 3
      EC(4) = 4
      pEC => EC
      c = 0
      write(*,*) 'cell',c,pEC
      call DMComplexSetCone(dm, c , pEC, ierr)
      CHKERRQ(ierr)
      call DMComplexGetCone(dm, c , pEC, ierr)
      CHKERRQ(ierr)
      write(*,*) 'cell',c,pEC

      call DMDestroy(dm,ierr)
      call PetscFinalize(ierr)
      end
