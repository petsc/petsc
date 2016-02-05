      program DMPlexTestField
      implicit none
#include "petsc/finclude/petsc.h90"
      DM :: dm
      DMLabel :: label
      Vec :: u
      PetscViewer :: viewer
      PetscSection :: section
      PetscInt :: dim,numFields,numBC,i,val
      PetscInt, target, dimension(3) ::                                 &
     &     numComp
      PetscInt, pointer :: pNumComp(:)
      PetscInt, target, dimension(12) ::                                &
     &     numDof
      PetscInt, pointer :: pNumDof(:)
      PetscInt, target, dimension(1) ::                                 &
     &     bcField
      PetscInt, pointer :: pBcField(:)
      IS, target, dimension(1) ::                                       &
     &     bcCompIS
      IS, target, dimension(1) ::                                       &
     &     bcPointIS
      IS, pointer :: pBcCompIS(:)
      IS, pointer :: pBcPointIS(:)
      PetscBool :: interpolate
      PetscErrorCode :: ierr

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      CHKERRQ(ierr)
      dim = 2
      call PetscOptionsGetInt(PETSC_NULL_OBJECT,PETSC_NULL_CHARACTER,   &
     &                        '-dim', dim,PETSC_NULL_BOOL, ierr)
      CHKERRQ(ierr)
      interpolate = PETSC_TRUE
!     Create a mesh
      call DMPlexCreateBoxMesh(PETSC_COMM_WORLD, dim, interpolate, dm,
     &     ierr)
      CHKERRQ(ierr)
!     Create a scalar field u, a vector field v, and a surface vector field w
      numFields  = 3
      numComp(1) = 1
      numComp(2) = dim
      numComp(3) = dim-1
      pNumComp => numComp
      do i = 1, numFields*(dim+1)
         numDof(i) = 0
      end do
!     Let u be defined on vertices
      numDof(0*(dim+1)+1)     = 1
!     Let v be defined on cells
      numDof(1*(dim+1)+dim+1) = dim
!     Let v be defined on faces
      numDof(2*(dim+1)+dim)   = dim-1
      pNumDof => numDof
!     Setup boundary conditions
      numBC = 1
!     Test label retrieval
      call DMGetLabel(dm, 'marker', label, ierr)
      CHKERRQ(ierr)
      call DMLabelGetValue(label, 0, val, ierr)
      CHKERRQ(ierr)
      if (val .ne. -1) then
        CHKERRQ(1)
      endif
      call DMLabelGetValue(label, 8, val, ierr)
      CHKERRQ(ierr)
      if (val .ne. 1) then
        CHKERRQ(1)
      endif
!     Prescribe a Dirichlet condition on u on the boundary
!       Label "marker" is made by the mesh creation routine
      bcField(1) = 0
      pBcField => bcField
      call ISCreateStride(PETSC_COMM_WORLD, 1, 0, 1, bcCompIS(1), ierr)
      CHKERRQ(ierr)
      pBcCompIS => bcCompIS
      call DMGetStratumIS(dm, 'marker', 1, bcPointIS(1),
     &    ierr)
      CHKERRQ(ierr)
      pBcPointIS => bcPointIS
!     Create a PetscSection with this data layout
      call DMPlexCreateSection(dm, dim, numFields, pNumComp,
     &     pNumDof, numBC, pBcField, pBcCompIS, pBcPointIS,
     &     PETSC_NULL_OBJECT, section, ierr)
      CHKERRQ(ierr)
      call ISDestroy(bcCompIS(1), ierr)
      CHKERRQ(ierr)
      call ISDestroy(bcPointIS(1), ierr)
      CHKERRQ(ierr)
!     Name the Field variables
      call PetscSectionSetFieldName(section, 0, 'u', ierr)
      CHKERRQ(ierr)
      call PetscSectionSetFieldName(section, 1, 'v', ierr)
      CHKERRQ(ierr)
      call PetscSectionSetFieldName(section, 2, 'w', ierr)
      CHKERRQ(ierr)
      call PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD, ierr)
      CHKERRQ(ierr)
!     Tell the DM to use this data layout
      call DMSetDefaultSection(dm, section, ierr)
      CHKERRQ(ierr)
!     Create a Vec with this layout and view it
      call DMGetGlobalVector(dm, u, ierr)
      CHKERRQ(ierr)
      call PetscViewerCreate(PETSC_COMM_WORLD, viewer, ierr)
      CHKERRQ(ierr)
      call PetscViewerSetType(viewer, PETSCVIEWERVTK, ierr)
      CHKERRQ(ierr)
      call PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_VTK, ierr)
      CHKERRQ(ierr)
      call PetscViewerFileSetName(viewer, 'sol.vtk', ierr)
      CHKERRQ(ierr)
      call VecView(u, viewer, ierr)
      CHKERRQ(ierr)
      call PetscViewerDestroy(viewer, ierr)
      CHKERRQ(ierr)
      call DMRestoreGlobalVector(dm, u, ierr)
      CHKERRQ(ierr)
!     Cleanup
      call PetscSectionDestroy(section, ierr)
      CHKERRQ(ierr)
      call DMDestroy(dm, ierr)
      CHKERRQ(ierr)

      call PetscFinalize(ierr)
      end program DMPlexTestField
