#include "petsc/finclude/petsc.h"
program ex47f90
  use petsc
  implicit none

  type(tDM)                         :: dm
  type(tPetscSection)               :: section
  character(len=PETSC_MAX_PATH_LEN) :: IOBuffer
  PetscInt                          :: dof, p, pStart, pEnd, d
  type(tVec)                        :: v
  PetscInt                          :: zero = 0
  PetscInt                          :: one = 1
  PetscInt                          :: two = 2
  PetscScalar, dimension(:), pointer  :: val
  PetscScalar, pointer              :: x(:)
  PetscErrorCode                    :: ierr

  PetscCallA(PetscInitialize(ierr))

  PetscCallA(DMCreate(PETSC_COMM_WORLD, dm, ierr))
  PetscCallA(DMSetType(dm, DMPLEX, ierr))
  PetscCallA(DMSetFromOptions(dm, ierr))
  PetscCallA(DMViewFromOptions(dm, PETSC_NULL_OBJECT, '-d_view', ierr))

  PetscCallA(PetscSectionCreate(PETSC_COMM_WORLD, section, ierr))
  PetscCallA(DMPlexGetChart(dm, pStart, pEnd, ierr))
  PetscCallA(PetscSectionSetChart(section, pStart, pEnd, ierr))
  PetscCallA(DMPlexGetHeightStratum(dm, zero, pStart, pEnd, ierr))
  do p = pStart, pEnd - 1
    PetscCallA(PetscSectionSetDof(section, p, one, ierr))
  end do
  PetscCallA(DMPlexGetDepthStratum(dm, zero, pStart, pEnd, ierr))
  do p = pStart, pEnd - 1
    PetscCallA(PetscSectionSetDof(section, p, two, ierr))
  end do
  PetscCallA(PetscSectionSetUp(section, ierr))
  PetscCallA(DMSetLocalSection(dm, section, ierr))
  PetscCallA(PetscSectionViewFromOptions(section, PETSC_NULL_OBJECT, '-s_view', ierr))

  PetscCallA(DMCreateGlobalVector(dm, v, ierr))

  PetscCallA(DMPlexGetChart(dm, pStart, pEnd, ierr))
  do p = pStart, pEnd - 1
    PetscCallA(PetscSectionGetDof(section, p, dof, ierr))
    allocate (val(dof))
    do d = 1, dof
      val(d) = 100*p + d - 1
    end do
    PetscCallA(VecSetValuesSection(v, section, p, val, INSERT_VALUES, ierr))
    deallocate (val)
  end do
  PetscCallA(VecView(v, PETSC_VIEWER_STDOUT_WORLD, ierr))

  do p = pStart, pEnd - 1
    PetscCallA(PetscSectionGetDof(section, p, dof, ierr))
    PetscCallA(VecGetValuesSection(v, section, p, x, ierr))
    write (IOBuffer, *) 'Point ', p, ' dof ', dof, '\n'
    PetscCallA(PetscPrintf(PETSC_COMM_SELF, IOBuffer, ierr))
    PetscCallA(VecRestoreValuesSection(v, section, p, x, ierr))
  end do

  PetscCallA(PetscSectionDestroy(section, ierr))
  PetscCallA(VecDestroy(v, ierr))
  PetscCallA(DMDestroy(dm, ierr))
  PetscCallA(PetscFinalize(ierr))
end program ex47f90

!/*TEST
!
!  test:
!    suffix: 0
!    args: -dm_plex_filename ${wPETSC_DIR}/share/petsc/datafiles/meshes/quads-q2.msh
!
!TEST*/
