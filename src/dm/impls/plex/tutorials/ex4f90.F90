! setting up DMPlex for finite elements
! Contributed by Pratheek Shanthraj <p.shanthraj@mpie.de>
program main
#include <petsc/finclude/petsc.h>
  use petsc
  implicit none
  DM :: dm
  PetscDS :: ds
  PetscInt :: dim = 3, zero = 0
  PetscBool :: simplex = PETSC_TRUE
  PetscBool :: interpolate = PETSC_TRUE
  PetscReal :: refinementLimit = 0.0
  PetscErrorCode :: ierr
  PetscTabulation, pointer :: tab(:)
  PetscFE fe, rfe
  PetscObject obj
  PetscInt :: one = 1, mone = -1

  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
  PetscCallA(DMPlexCreateDoublet(PETSC_COMM_WORLD, dim, simplex, interpolate, refinementLimit, dm, ierr))
  PetscCallA(PetscFECreateDefault(PETSC_COMM_WORLD, dim, one, simplex, 'name', mone, fe, ierr))
  PetscCallA(PetscObjectSetName(fe, 'name', ierr))
  PetscCallA(DMSetField(dm, zero, PETSC_NULL_DMLABEL, PetscObjectCast(fe), ierr))
  PetscCallA(DMSetField(dm, one, PETSC_NULL_DMLABEL, PetscObjectCast(fe), ierr))

  PetscCallA(DMSetUp(dm, ierr))
  PetscCallA(DMCreateDS(dm, ierr))
  PetscCallA(DMGetDS(dm, ds, ierr))
  PetscCallA(PetscDSGetTabulation(ds, tab, ierr))
  print *, tab(1)%ptr%T(1)%ptr
  print *, tab(1)%ptr%T(2)%ptr
  print *, tab(2)%ptr%T(1)%ptr
  print *, tab(2)%ptr%T(2)%ptr
  PetscCallA(PetscDSRestoreTabulation(ds, tab, ierr))

  PetscCallA(PetscDSGetDiscretization(ds, zero, obj, ierr))
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
