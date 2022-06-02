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
      PetscMPIInt :: size
      IS, target, dimension(1) ::   bcCompIS
      IS, target, dimension(1) ::   bcPointIS
      IS, pointer :: pBcCompIS(:)
      IS, pointer :: pBcPointIS(:)
      PetscErrorCode :: ierr

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))
!     Create a mesh
      PetscCallA(DMCreate(PETSC_COMM_WORLD, dm, ierr))
      PetscCallA(DMSetType(dm, DMPLEX, ierr))
      PetscCallA(DMSetFromOptions(dm, ierr))
      PetscCallA(DMViewFromOptions(dm, PETSC_NULL_VEC, '-dm_view', ierr))
      PetscCallA(DMGetDimension(dm, dim, ierr))
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
      PetscCallA(DMGetLabel(dm, 'marker', label, ierr))
      PetscCallA(DMLabelGetValue(label, zero, val, ierr))
      if (size .eq. 1 .and. val .ne. -1) then
        SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error in library')
      endif
      PetscCallA(DMLabelGetValue(label, eight, val, ierr))
      if (size .eq. 1 .and. val .ne. 1) then
        SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error in library')
      endif
!     Prescribe a Dirichlet condition on u on the boundary
!       Label "marker" is made by the mesh creation routine
      bcField(1) = 0
      pBcField => bcField
      PetscCallA(ISCreateStride(PETSC_COMM_WORLD, one, zero, one, bcCompIS(1), ierr))
      pBcCompIS => bcCompIS
      PetscCallA(DMGetStratumIS(dm, 'marker', one, bcPointIS(1),ierr))
      pBcPointIS => bcPointIS
!     Create a PetscSection with this data layout
      PetscCallA(DMSetNumFields(dm, numFields,ierr))
      PetscCallA(DMPlexCreateSection(dm,nolabel,pNumComp,pNumDof,numBC,pBcField,pBcCompIS,pBcPointIS,PETSC_NULL_IS,section,ierr))
      PetscCallA(ISDestroy(bcCompIS(1), ierr))
      PetscCallA(ISDestroy(bcPointIS(1), ierr))
!     Name the Field variables
      PetscCallA(PetscSectionSetFieldName(section, zero, 'u', ierr))
      PetscCallA(PetscSectionSetFieldName(section, one,  'v', ierr))
      PetscCallA(PetscSectionSetFieldName(section, two,  'w', ierr))
      if (size .eq. 1) then
        PetscCallA(PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD, ierr))
      endif
!     Tell the DM to use this data layout
      PetscCallA(DMSetLocalSection(dm, section, ierr))
!     Create a Vec with this layout and view it
      PetscCallA(DMGetGlobalVector(dm, u, ierr))
      PetscCallA(PetscViewerCreate(PETSC_COMM_WORLD, viewer, ierr))
      PetscCallA(PetscViewerSetType(viewer, PETSCVIEWERVTK, ierr))
      PetscCallA(PetscViewerFileSetName(viewer, 'sol.vtu', ierr))
      PetscCallA(VecView(u, viewer, ierr))
      PetscCallA(PetscViewerDestroy(viewer, ierr))
      PetscCallA(DMRestoreGlobalVector(dm, u, ierr))
!     Cleanup
      PetscCallA(PetscSectionDestroy(section, ierr))
      PetscCallA(DMDestroy(dm, ierr))

      PetscCallA(PetscFinalize(ierr))
      end program DMPlexTestField

!/*TEST
!  build:
!    requires: defined(PETSC_USING_F90FREEFORM)
!
!  test:
!    suffix: 0
!    requires: triangle
!    args: -info :~sys,mat:
!
!  test:
!    suffix: 0_2
!    requires: triangle
!    nsize: 2
!    args: -info :~sys,mat,partitioner:
!
!  test:
!    suffix: 1
!    requires: ctetgen
!    args: -dm_plex_dim 3 -info :~sys,mat:
!
!TEST*/
