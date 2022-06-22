! setting up DMPlex for finite elements
! Contributed by Pratheek Shanthraj <p.shanthraj@mpie.de>
      program main
      implicit none
#include <petsc/finclude/petsc.h90>
      DM :: dm
      PetscDS :: prob
      PetscInt :: dim = 3
      PetscBool :: simplex = PETSC_TRUE
      PetscBool :: interpolate = PETSC_TRUE
      PetscBool :: refinementUniform = PETSC_FALSE
      PetscReal :: refinementLimit = 0.0
      PetscErrorCode :: ierr

      PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
      PetscCallA(DMPlexCreateDoublet(PETSC_COMM_WORLD, dim, simplex,interpolate, refinementUniform, refinementLimit, dm, ierr))
      PetscCallA(DMSetUp(dm,ierr))
      PetscCallA(PetscDSCreate(PETSC_COMM_WORLD,prob,ierr))
      PetscCallA(DMGetDS(dm,prob,ierr))

      PetscCallA(DMDestroy(dm, ierr))
      PetscCallA(PetscFinalize(ierr))
      end program main
