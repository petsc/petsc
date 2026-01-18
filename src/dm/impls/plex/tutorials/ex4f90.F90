! setting up DMPlex for finite elements
! Contributed by Pratheek Shanthraj <p.shanthraj@mpie.de>
#include <petsc/finclude/petsc.h>
program main
  use petsc
  implicit none
  DM :: dm
  PetscDS :: ds
  PetscInt, parameter :: dim = 3
  PetscBool, parameter :: simplex = PETSC_TRUE, interpolate = PETSC_TRUE
  PetscReal, parameter :: refinementLimit = 0.0
  PetscErrorCode :: ierr
  PetscTabulation, pointer :: tab(:)
  PetscFE fe, rfe
  PetscObject obj

  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
  PetscCallA(DMPlexCreateDoublet(PETSC_COMM_WORLD, dim, simplex, interpolate, refinementLimit, dm, ierr))
  PetscCallA(PetscFECreateDefault(PETSC_COMM_WORLD, dim, 1_PETSC_INT_KIND, simplex, 'name', -1_PETSC_INT_KIND, fe, ierr))
  PetscCallA(PetscObjectSetName(fe, 'name', ierr))
  PetscCallA(DMSetField(dm, 0_PETSC_INT_KIND, PETSC_NULL_DMLABEL, PetscObjectCast(fe), ierr))
  PetscCallA(DMSetField(dm, 1_PETSC_INT_KIND, PETSC_NULL_DMLABEL, PetscObjectCast(fe), ierr))

  PetscCallA(DMSetUp(dm, ierr))
  PetscCallA(DMCreateDS(dm, ierr))
  PetscCallA(DMGetDS(dm, ds, ierr))
  PetscCallA(PetscDSGetTabulation(ds, tab, ierr))
  print *, tab(1)%ptr%T(1)%ptr
  print *, tab(1)%ptr%T(2)%ptr
  print *, tab(2)%ptr%T(1)%ptr
  print *, tab(2)%ptr%T(2)%ptr
  PetscCallA(PetscDSRestoreTabulation(ds, tab, ierr))

  PetscCallA(PetscDSGetDiscretization(ds, 0_PETSC_INT_KIND, obj, ierr))
  PetscObjectSpecificCast(rfe, obj)
  PetscCallA(PetscFEDestroy(fe, ierr))
  PetscCallA(DMDestroy(dm, ierr))
  PetscCallA(PetscFinalize(ierr))
end program main
!/*TEST
!
!  test:
!    nsize: 1
!
!TEST*/
