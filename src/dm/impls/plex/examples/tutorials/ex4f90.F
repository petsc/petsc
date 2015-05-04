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

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      CHKERRQ(ierr)
      call DMPlexCreateDoublet(PETSC_COMM_WORLD, dim, simplex,          &
     &     interpolate, refinementUniform, refinementLimit, dm, ierr)
      CHKERRQ(ierr)
      call DMSetUp(dm,ierr)
      CHKERRQ(ierr)
      call PetscDSCreate(PETSC_COMM_WORLD,prob,ierr)
      CHKERRQ(ierr)
      call DMGetDS(dm,prob,ierr)
      CHKERRQ(ierr)

      call DMDestroy(dm, ierr)
      CHKERRQ(ierr)
      call PetscFinalize(ierr)
      CHKERRQ(ierr)
      end program main
