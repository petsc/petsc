      program main
!
#include <finclude/petscsysdef.h>
#include <finclude/petscvecdef.h>
#include <finclude/petscmatdef.h>
#include <finclude/petscdmmeshdef.h>
      use petscsys
      use petscvec
      use petscmat
      use petscdmmesh
!
      type(DM)  mesh
      type(Mat) A
      type(PetscSection) section
      PetscInt  dim
      PetscInt  numDof(3)

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      CHKERRQ(ierr)
!     Create mesh from an ExodusII file
      call DMMeshCreateExodus(PETSC_COMM_WORLD, 'Square.gen', mesh,     &
     &     ierr)
      CHKERRQ(ierr)
      call DMSetFromOptions(mesh,ierr)
      CHKERRQ(ierr)
      call DMView(mesh,PETSC_VIEWER_STDOUT_WORLD,ierr)
      CHKERRQ(ierr)
!     Create dof layout
      call DMMeshGetDimension(mesh,dim,ierr)
      CHKERRQ(ierr)
      numDof(1) = 1
      numDof(2) = 0
      numDof(3) = 0
      call DMMeshCreateSection(mesh,dim,numDof,PETSC_NULL_CHARACTER,1,  &
     &     section,ierr)
      CHKERRQ(ierr)
      call DMMeshSetSection(mesh,'default',section,ierr)
      CHKERRQ(ierr)
!     Create preallocated matrix
      call DMCreateMatrix(mesh, MATAIJ, A,ierr)
      CHKERRQ(ierr)
      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr)
      CHKERRQ(ierr)
!     Cleanup
      call MatDestroy(A,ierr)
      CHKERRQ(ierr)
      call DMDestroy(mesh,ierr)
      CHKERRQ(ierr)
      call PetscFinalize(ierr)
      end
