!   Use DMPlexGetClosureIndices to check for shared node DOF
!   The mesh consists of two tetrehadra, sharing a (triangular) face, hence
!   the number of shared DOF equals 3 nodes x 3 dof/node = 9
!
program main
#include <petsc/finclude/petscdmplex.h>
#include <petsc/finclude/petscdm.h>
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petsc.h>

  use PETScDM
  use PETScDMplex

  implicit none

  DM :: dm, cdm
  PetscInt :: cStart, cEnd
  PetscInt :: cdim, nIdx, idx, cnt, Nf
  PetscInt, parameter :: sharedNodes = 3, zero = 0
  PetscSection :: gS
  PetscErrorCode :: ierr

  PetscInt, allocatable :: idxMatrix(:, :), offsets(:)
  PetscInt, pointer, dimension(:) :: indices

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(DMCreate(PETSC_COMM_WORLD, dm, ierr))
  PetscCallA(DMSetType(dm, DMPLEX, ierr))
  PetscCallA(DMSetFromOptions(dm, ierr))

  PetscCallA(DMGetCoordinateDM(dm, cdm, ierr))
  PetscCallA(DMGetCoordinateDim(cdm, cdim, ierr))
  PetscCallA(DMGetGlobalSection(cdm, gS, ierr))

  PetscCallA(DMPlexGetHeightStratum(dm, zero, cStart, cEnd, ierr))
  PetscCallA(PetscSectionGetNumFields(gS, Nf, ierr))
  allocate (offsets(Nf + 1), source=zero)

  ! Indices per cell
  ! cell 0 (cStart)
  PetscCallA(DMPlexGetClosureIndices(cdm, gS, gS, cStart, PETSC_TRUE, nIdx, indices, offsets, PETSC_NULL_SCALAR_POINTER, ierr))
  allocate (idxMatrix(nIdx, cEnd - cStart))
  idxMatrix(1:nIdx, cStart + 1) = indices
  ! Check size and content of output field offsets array
  PetscCheckA(size(offsets) == (Nf + 1) .and. offsets(1) == zero, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Wrong field offsets")
  PetscCallA(DMPlexRestoreClosureIndices(cdm, gS, gS, cStart, PETSC_TRUE, nIdx, indices, offsets, PETSC_NULL_SCALAR_POINTER, ierr))
  ! cell 1 (cEnd - 1)
  PetscCallA(DMPlexGetClosureIndices(cdm, gS, gS, cEnd - 1, PETSC_TRUE, nIdx, indices, offsets, PETSC_NULL_SCALAR_POINTER, ierr))
  idxMatrix(1:nIdx, cEnd - Cstart) = indices
  PetscCallA(DMPlexRestoreClosureIndices(cdm, gS, gS, cEnd - 1, PETSC_TRUE, nIdx, indices, offsets, PETSC_NULL_SCALAR_POINTER, ierr))

  ! Check number of shared indices between cell 0 and cell 1
  cnt = 0
  do idx = 1, nIdx
    cnt = cnt + count(idxMatrix(idx, 1) == idxMatrix(1:nIdx, cEnd))
  end do
  PetscCheckA(cnt == sharedNodes*cdim, PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Wrong DOF indices")

  ! Cleanup
  deallocate (offsets)
  deallocate (idxMatrix)
  PetscCallA(DMDestroy(dm, ierr))
  PetscCallA(PetscFinalize(ierr))

end program main

! /*TEST
!
! test:
!   args : -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/doublet-tet.msh
!   output_file: output/empty.out
!
! TEST*/
