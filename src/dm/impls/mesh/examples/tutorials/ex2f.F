      program main
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscdm.h>
#include <finclude/petscdmmesh.h>

      PetscInt       cells(24)
      data cells/0, 3, 1,  3, 4, 1,  4, 2, 1,  4, 5, 2,  6, 4, 3,       &
     &  6, 7, 4,  7, 5, 4,  7, 8, 5/
      PetscReal      coords(18)
      data coords/0., 0., 0.5, 0., 1., 0., 0., 0.5, 0.5, 0.5, 1., 0.5,  &
     &     0., 1., 0.5, 1., 1., 1./
      DM             dm, distributedDM
      PetscInt       numCells, numVertices
      PetscMPIInt    rank
      PetscErrorCode ierr

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      CHKERRQ(ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      CHKERRQ(ierr)
      if (rank.eq.0) then
         numCells    = 8
         numVertices = 9
      else
         numCells    = 0
         numVertices = 0
      endif
      call DMMeshCreateMeshFromAdjacency(PETSC_COMM_WORLD, 2, numCells, &
     &     3, cells, 2, numVertices, coords, PETSC_FALSE, dm, ierr)
      CHKERRQ(ierr)
      call DMSetFromOptions(dm, ierr)
      CHKERRQ(ierr)
c     Distribute mesh over processes
      distributedDM = PETSC_NULL;
      call DMMeshDistribute(dm, PETSC_NULL_CHARACTER, distributedDM,    &
     &     ierr)
      CHKERRQ(ierr)
      if (distributedDM.ne.PETSC_NULL) then
        call DMDestroy(dm, ierr)
        CHKERRQ(ierr)
        dm = distributedDM
        call DMSetFromOptions(dm, ierr)
        CHKERRQ(ierr)
      endif
      call DMDestroy(dm, ierr)
      CHKERRQ(ierr)
      call PetscFinalize(ierr)
      end
