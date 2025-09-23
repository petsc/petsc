program ex14f90

#include <petsc/finclude/petsc.h>
  use petsc
  use mpi     ! needed when PETSC_HAVE_MPI_F90MODULE is not true to define MPI_REPLACE
  implicit none

  type(tDM)                        :: dm
  type(tVec)                       :: u
  type(tPetscSection)              :: section
  PetscInt                         :: dim, numFields, numBC
  PetscMPIInt                      :: rank
  PetscInt, dimension(2)            :: numComp
  PetscInt, dimension(12)           :: numDof
  PetscInt, dimension(:), pointer    :: remoteOffsets
  type(tPetscSF)                   :: pointSF
  type(tPetscSF)                   :: sectionSF
  PetscScalar, dimension(:), pointer :: array
  PetscReal                        :: val
  PetscErrorCode                   :: ierr
  PetscInt                         :: zero = 0, one = 1

  PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER, ierr))
  PetscCallA(DMCreate(PETSC_COMM_WORLD, dm, ierr))
  PetscCallA(DMSetType(dm, DMPLEX, ierr))
  PetscCallA(DMSetFromOptions(dm, ierr))
  PetscCallA(DMViewFromOptions(dm, PETSC_NuLL_OBJECT, "-dm_view", ierr))
  PetscCallA(DMGetDimension(dm, dim, ierr))

  !!! Describe the solution variables that are discretized on the mesh
  ! Create scalar field u and a vector field v
  numFields = 2
  numComp = [one, dim]
  numDof = 0
  !Let u be defined on cells
  numDof(0*(dim + 1) + dim + 1) = 1
  !Let v be defined on vertices
  numDof(1*(dim + 1) + 1) = dim
  !No boundary conditions */
  numBC = 0

  !!! Create a PetscSection to handle the layout of the discretized variables
  PetscCallA(DMSetNumFields(dm, numFields, ierr))
  PetscCallA(DMPlexCreateSection(dm, PETSC_NULL_DMLABEL_ARRAY, numComp, numDof, numBC, PETSC_NULL_INTEGER_ARRAY, PETSC_NULL_IS_ARRAY, PETSC_NULL_IS_ARRAY, PETSC_NULL_IS, section, ierr))
  ! Name the Field variables
  PetscCallA(PetscSectionSetFieldName(section, zero, "u", ierr))
  PetscCallA(PetscSectionSetFieldName(section, one, "v", ierr))
  ! Tell the DM to use this data layout
  PetscCallA(DMSetLocalSection(dm, section, ierr))

  !!! Construct the communication pattern for halo exchange between local vectors */
  ! Get the point SF: an object that says which copies of mesh points (cells,
  ! vertices, faces, edges) are copies of points on other processes
  PetscCallA(DMGetPointSF(dm, pointSF, ierr))
  ! Relate the locations of ghost degrees of freedom on this process
  ! to their locations of the non-ghost copies on a different process
  PetscCallA(PetscSFCreateRemoteOffsets(pointSF, section, section, remoteOffsets, ierr))
  ! Use that information to construct a star forest for halo exchange
  ! for data described by the local section
  PetscCallA(PetscSFCreateSectionSF(pointSF, section, remoteOffsets, section, sectionSF, ierr))
  if (associated(remoteOffsets)) then
    PetscCallA(PetscSFDestroyRemoteOffsets(remoteOffsets, ierr))
  end if

  !!! Demo of halo exchange
  ! Create a Vec with this layout
  PetscCallA(DMCreateLocalVector(dm, u, ierr))
  PetscCallA(PetscObjectSetName(u, "Local vector", ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))
  ! Set all mesh values to the MPI rank
  val = rank
  PetscCallA(VecSet(u, val, ierr))
  ! Get the raw array of values
  PetscCallA(VecGetArrayWrite(u, array, ierr))
  !!! HALO EXCHANGE
  PetscCallA(PetscSFBcastBegin(sectionSF, MPIU_SCALAR, array, array, MPI_REPLACE, ierr))
  ! local work can be done between Begin() and End()
  PetscCallA(PetscSFBcastEnd(sectionSF, MPIU_SCALAR, array, array, MPI_REPLACE, ierr))
  ! Restore the raw array of values
  PetscCallA(VecRestoreArrayWrite(u, array, ierr))
  ! View the results: should show which process has the non-ghost copy of each degree of freedom
  PetscCallA(PetscSectionVecView(section, u, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(VecDestroy(u, ierr))

  PetscCallA(PetscSFDestroy(sectionSF, ierr))
  PetscCallA(PetscSectionDestroy(section, ierr))
  PetscCallA(DMDestroy(dm, ierr))
  PetscCallA(PetscFinalize(ierr))
end program ex14f90
!/*TEST
!  build:
!    requires: defined(PETSC_USING_F90FREEFORM)
!
!  # Test on a 1D mesh with overlap
!  test:
!    nsize: 3
!    requires: !complex
!    args: -dm_plex_dim 1 -dm_plex_box_faces 3 -dm_refine_pre 1 -petscpartitioner_type simple -dm_distribute_overlap 1
!
!TEST*/
