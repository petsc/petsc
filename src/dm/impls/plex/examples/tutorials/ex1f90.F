      program DMPlexTestField
      implicit none
#include "finclude/petsc.h"
      DM             :: dm
      PetscSection   :: section
      PetscInt       :: dim,numFields,numBC,i
      PetscInt       :: numComp(3)
      PetscInt       :: numDof(12)
      PetscInt       :: bcField(1)
      IS             :: bcPointIS(1)
      PetscBool      :: interpolate
      PetscErrorCode :: ierr

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      CHKERRQ(ierr)
      dim = 2
      call PetscOptionsGetInt(PETSC_NULL_CHARACTER, "-dim", dim,
     $     PETSC_NULL_BOOL, ierr)
      CHKERRQ(ierr)
      interpolate = PETSC_TRUE
!     Create a mesh
      call DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, interpolate, dm,
     $     ierr)
      CHKERRQ(ierr)
!     Create a scalar field u, a vector field v, and a surface vector field w
      numFields  = 3
      numComp(1) = 1
      numComp(2) = dim
      numComp(3) = dim-1
      do i = 1, numFields*(dim+1)
         numDof(i) = 0
      end do
!     Let u be defined on vertices
      numDof(0*(dim+1)+1)     = 1
!     Let v be defined on cells
      numDof(1*(dim+1)+dim+1) = dim
!     Let v be defined on faces
      numDof(2*(dim+1)+dim)   = dim-1
!     Setup boundary conditions
      numBC = 1
!     Prescribe a Dirichlet condition on u on the boundary
!       Label "marker" is made by the mesh creation routine
      bcField(1) = 0
      call DMPlexGetStratumIS(dm, "marker", 1, bcPointIS(1),
     $     ierr)
      CHKERRQ(ierr)
!     Create a PetscSection with this data layout
      call DMPlexCreateSection(dm, dim, numFields, numComp,
     $     numDof, numBC, bcField, bcPointIS, section,
     $     ierr)
      CHKERRQ(ierr)
!     Name the Field variables
      call PetscSectionSetFieldName(section, 0, "u", ierr)
      CHKERRQ(ierr)
      call PetscSectionSetFieldName(section, 1, "v", ierr)
      CHKERRQ(ierr)
      call PetscSectionSetFieldName(section, 2, "w", ierr)
      CHKERRQ(ierr)
      call PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD, ierr)
      CHKERRQ(ierr)
!     Tell the DM to use this data layout
      call DMSetDefaultSection(dm, section, ierr)
      CHKERRQ(ierr)
!     Cleanup
      call PetscSectionDestroy(section, ierr)
      CHKERRQ(ierr)
      call DMDestroy(dm, ierr)
      CHKERRQ(ierr)

      call PetscFinalize(ierr)
      end program DMPlexTestField
