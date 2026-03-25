#include <petsc/finclude/petscdmplex.h>
#include <petsc/finclude/petscdmlabel.h>
program DMPlexTestField
  use petscdm
  use petscdmplex
  implicit none

  DM :: dm
  DMLabel :: label
  Vec :: u
  PetscViewer :: viewer
  PetscSection :: section
  PetscInt, parameter :: numFields = 3, numBC = 1
  PetscInt :: dim, val
  PetscInt, target, dimension(3) ::  numComp
  PetscInt, pointer :: pNumComp(:)
  PetscInt, target, dimension(12) ::  numDof
  PetscInt, pointer :: pNumDof(:)
  PetscInt, target, dimension(1) ::  bcField
  PetscInt, pointer :: pBcField(:)
  PetscMPIInt :: size
  IS, target, dimension(1) ::   bcCompIS
  IS, target, dimension(1) ::   bcPointIS
  IS, pointer :: pBcCompIS(:)
  IS, pointer :: pBcPointIS(:)
  PetscErrorCode :: ierr

  PetscCallA(PetscInitialize(ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))
! Create a mesh
  PetscCallA(DMCreate(PETSC_COMM_WORLD, dm, ierr))
  PetscCallA(DMSetType(dm, DMPLEX, ierr))
  PetscCallA(DMSetFromOptions(dm, ierr))
  PetscCallA(DMViewFromOptions(dm, PETSC_NULL_OBJECT, '-dm_view', ierr))
  PetscCallA(DMGetDimension(dm, dim, ierr))
! Create a scalar field u, a vector field v, and a surface vector field w
  numComp = [1_PETSC_INT_KIND, dim, dim - 1_PETSC_INT_KIND]
  pNumComp => numComp
  numDof(1:numFields*(dim + 1)) = 0
! Let u be defined on vertices
  numDof(0*(dim + 1) + 1) = 1
! Let v be defined on cells
  numDof(1*(dim + 1) + dim + 1) = dim
! Let v be defined on faces
  numDof(2*(dim + 1) + dim) = dim - 1
  pNumDof => numDof
! Test label retrieval
  PetscCallA(DMGetLabel(dm, 'marker', label, ierr))
  PetscCallA(DMLabelGetValue(label, 0_PETSC_INT_KIND, val, ierr))
  PetscCheckA(size /= 1 .or. val == -1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Error in library')
  PetscCallA(DMLabelGetValue(label, 8_PETSC_INT_KIND, val, ierr))
  PetscCheckA(size /= 1 .or. val == 1, PETSC_COMM_SELF, PETSC_ERR_PLIB, 'Error in library')
! Prescribe a Dirichlet condition on u on the boundary
! Label "marker" is made by the mesh creation routine
  bcField(1) = 0
  pBcField => bcField
  PetscCallA(ISCreateStride(PETSC_COMM_WORLD, 1_PETSC_INT_KIND, 0_PETSC_INT_KIND, 1_PETSC_INT_KIND, bcCompIS(1), ierr))
  pBcCompIS => bcCompIS
  PetscCallA(DMGetStratumIS(dm, 'marker', 1_PETSC_INT_KIND, bcPointIS(1), ierr))
  pBcPointIS => bcPointIS
! Create a PetscSection with this data layout
  PetscCallA(DMSetNumFields(dm, numFields, ierr))
  PetscCallA(DMPlexCreateSection(dm, PETSC_NULL_DMLABEL_ARRAY, pNumComp, pNumDof, numBC, pBcField, pBcCompIS, pBcPointIS, PETSC_NULL_IS, section, ierr))
  PetscCallA(ISDestroy(bcCompIS(1), ierr))
  PetscCallA(ISDestroy(bcPointIS(1), ierr))
! Name the Field variables
  PetscCallA(PetscSectionSetFieldName(section, 0_PETSC_INT_KIND, 'u', ierr))
  PetscCallA(PetscSectionSetFieldName(section, 1_PETSC_INT_KIND, 'v', ierr))
  PetscCallA(PetscSectionSetFieldName(section, 2_PETSC_INT_KIND, 'w', ierr))
  if (size == 1) then
    PetscCallA(PetscSectionView(section, PETSC_VIEWER_STDOUT_WORLD, ierr))
  end if
! Tell the DM to use this data layout
  PetscCallA(DMSetLocalSection(dm, section, ierr))
! Create a Vec with this layout and view it
  PetscCallA(DMGetGlobalVector(dm, u, ierr))
  PetscCallA(PetscViewerCreate(PETSC_COMM_WORLD, viewer, ierr))
  PetscCallA(PetscViewerSetType(viewer, PETSCVIEWERVTK, ierr))
  PetscCallA(PetscViewerFileSetName(viewer, 'sol.vtu', ierr))
  PetscCallA(VecView(u, viewer, ierr))
  PetscCallA(PetscViewerDestroy(viewer, ierr))
  PetscCallA(DMRestoreGlobalVector(dm, u, ierr))
! Cleanup
  PetscCallA(PetscSectionDestroy(section, ierr))
  PetscCallA(DMDestroy(dm, ierr))

  PetscCallA(PetscFinalize(ierr))
end program DMPlexTestField

!/*TEST
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
