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
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call DMPlexCreateDoublet(PETSC_COMM_WORLD, dim, simplex,interpolate, refinementUniform, refinementLimit, dm, ierr);CHKERRA(ierr)
      call DMSetUp(dm,ierr);CHKERRA(ierr)
      call PetscDSCreate(PETSC_COMM_WORLD,prob,ierr);CHKERRA(ierr)
      call DMGetDS(dm,prob,ierr);CHKERRA(ierr)

      call DMDestroy(dm, ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end program main
