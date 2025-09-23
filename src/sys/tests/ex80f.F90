!
! PETSc Program to test HDF5 viewer and HDF5 attribute I/O
!
program main
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petscvec.h>
  use petscsys
  use petscvec
  implicit none

  PetscViewer :: viewer
  PetscErrorCode :: ierr
  Vec :: x
  PetscReal, parameter :: one = 1.0
  PetscInt :: ival = 42
  PetscReal :: rval = 3.14
  ! initialize PETSc
  PetscCallA(PetscInitialize(ierr))
  ! create and write a vector
  PetscCallA(VecCreate(PETSC_COMM_WORLD, x, ierr))
  PetscCallA(PetscObjectSetName(x, "vec", ierr))
  PetscCallA(VecSetSizes(x, 3, PETSC_DETERMINE, ierr))
  PetscCallA(VecSetType(x, VECSTANDARD, ierr))
  PetscCallA(VecSet(x, one, ierr))
  PetscCallA(PetscViewerCreate(PETSC_COMM_WORLD, viewer, ierr))
  PetscCallA(PetscViewerSetType(viewer, PETSCVIEWERHDF5, ierr))
  PetscCallA(PetscViewerFileSetMode(viewer, FILE_MODE_WRITE, ierr))
  PetscCallA(PetscViewerFileSetName(viewer, "ex80f.hdf5", ierr))
  PetscCallA(VecView(x, viewer, ierr))
  PetscCallA(PetscViewerHDF5WriteAttribute(viewer, "vec", "int_attribute", ival, ierr))
  PetscCallA(PetscViewerHDF5WriteAttribute(viewer, "vec", "float_attribute", rval, ierr))
  PetscCallA(PetscViewerDestroy(viewer, ierr))
  PetscCallA(VecDestroy(x, ierr))
  PetscCallA(PetscFinalize(ierr))
end program main
!/*TEST
!  build:
!    requires: hdf5
!TEST*/