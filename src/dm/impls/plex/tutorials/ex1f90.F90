      program DMPlexTestField
#include "petsc/finclude/petscdmplex.h"
#include "petsc/finclude/petscdmlabel.h"
      use petscdmplex
      use petscsys
      implicit none

      DM :: dm
      DMLabel :: label
      Vec :: u
      PetscViewer :: viewer
      PetscSection :: section
      PetscInt :: dim,numFields,numBC
      PetscInt :: i,val
      DMLabel, pointer :: nolabel(:) => NULL()
      PetscInt, target, dimension(3) ::  numComp
      PetscInt, pointer :: pNumComp(:)
      PetscInt, target, dimension(12) ::  numDof
      PetscInt, pointer :: pNumDof(:)
      PetscInt, target, dimension(1) ::  bcField
      PetscInt, pointer :: pBcField(:)
      PetscInt, parameter :: zero = 0, one = 1, two = 2, eight = 8
      IS, target, dimension(1) ::   bcCompIS
      IS, target, dimension(1) ::   bcPointIS
      IS, pointer :: pBcCompIS(:)
      IS, pointer :: pBcPointIS(:)
      PetscErrorCode :: ierr

      call PetscInitialize(PETSC_NULL_CHARACTER, ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
!     Create a mesh
      call DMCreate(PETSC_COMM_WORLD, dm, ierr);CHKERRA(ierr)
      call DMSetType(dm, DMPLEX, ierr);CHKERRA(ierr)
      call DMSetFromOptions(dm, ierr);CHKERRA(ierr)
      call DMGetDimension(dm, dim, ierr);CHKERRA(ierr)
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
      call DMGetLabel(dm, 'marker', label, ierr);CHKERRA(ierr)
      call DMLabelGetValue(label, zero, val, ierr);CHKERRA(ierr)
      if (val .ne. -1) then
        SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error in library')
      endif
      call DMLabelGetValue(label, eight, val, ierr);CHKERRA(ierr)
      if (val .ne. 1) then
        SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error in library')
      endif
!     Prescribe a Dirichlet condition on u on the boundary
!       Label "marker" is made by the mesh creation routine
      bcField(1) = 0
      pBcField => bcField
      call ISCreateStride(PETSC_COMM_WORLD, one, zero, one, bcCompIS(1), ierr);CHKERRA(ierr)
      pBcCompIS => bcCompIS
      call DMGetStratumIS(dm, 'marker', one, bcPointIS(1),ierr);CHKERRA(ierr)
      pBcPointIS => bcPointIS
!     Create a PetscSection with this data layout
      call DMSetNumFields(dm, numFields,ierr);CHKERRA(ierr)
      call DMPlexCreateSection(dm,nolabel,pNumComp,pNumDof,numBC,pBcField,pBcCompIS,pBcPointIS,PETSC_NULL_IS,section,ierr)
      CHKERRA(ierr)
      call ISDestroy(bcCompIS(1), ierr);CHKERRA(ierr)
      call ISDestroy(bcPointIS(1), ierr);CHKERRA(ierr)
!     Name the Field variables
      call PetscSectionSetFieldName(section, zero, 'u', ierr);CHKERRA(ierr)
      call PetscSectionSetFieldName(section, one,  'v', ierr);CHKERRA(ierr)
      call PetscSectionSetFieldName(section, two,  'w', ierr);CHKERRA(ierr)
      call PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD, ierr);CHKERRA(ierr)
!     Tell the DM to use this data layout
      call DMSetLocalSection(dm, section, ierr);CHKERRA(ierr)
!     Create a Vec with this layout and view it
      call DMGetGlobalVector(dm, u, ierr);CHKERRA(ierr)
      call PetscViewerCreate(PETSC_COMM_WORLD, viewer, ierr);CHKERRA(ierr)
      call PetscViewerSetType(viewer, PETSCVIEWERVTK, ierr);CHKERRA(ierr)
      call PetscViewerFileSetName(viewer, 'sol.vtu', ierr);CHKERRA(ierr)
      call VecView(u, viewer, ierr);CHKERRA(ierr)
      call PetscViewerDestroy(viewer, ierr);CHKERRA(ierr)
      call DMRestoreGlobalVector(dm, u, ierr);CHKERRA(ierr)
!     Cleanup
      call PetscSectionDestroy(section, ierr);CHKERRA(ierr)
      call DMDestroy(dm, ierr);CHKERRA(ierr)

      call PetscFinalize(ierr)
      end program DMPlexTestField

!/*TEST
!  build:
!    requires: define(PETSC_USING_F90FREEFORM)
!
!  test:
!    suffix: 0
!    requires: triangle
!    args: -info :~sys,mat:
!
!  test:
!    suffix: 1
!    requires: ctetgen
!    args: -dm_plex_dim 3 -info :~sys,mat:
!
!TEST*/
