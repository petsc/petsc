!    Description: A star forest is a simple tree with one root and zero or more leaves.
!    Many common communication patterns can be expressed as updates of rootdata using leafdata and vice-versa.
!     This example creates a star forest, communicates values using the graph  views the graph, then destroys it.
!
!     This is a copy of ex1.c but currently only tests the broadcast operation

program main
#include <petsc/finclude/petscvec.h>
  use petscmpi  ! or mpi or mpi_f08
  use petscvec
  implicit none

  PetscErrorCode ierr
  PetscInt i, nroots, nrootsalloc, nleaves, nleavesalloc, mine(6), stride
  PetscSFNode remote(6)
  PetscMPIInt rank, size
  PetscSF sf
  PetscInt rootdata(6), leafdata(6)

! used with PetscSFGetGraph()
  PetscSFNode, pointer ::       gremote(:)
  PetscInt, pointer ::          gmine(:)
  PetscInt gnroots, gnleaves

  PetscMPIInt niranks, nranks
  PetscMPIInt, pointer ::       iranks(:), ranks(:)
  PetscInt, pointer ::          ioffset(:), irootloc(:), roffset(:), rmine(:), rremote(:)

  PetscCallA(PetscInitialize(ierr))
  stride = 2
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD, size, ierr))

  if (rank == 0) then
    nroots = 3
  else
    nroots = 2
  end if
  nrootsalloc = nroots*stride
  if (rank > 0) then
    nleaves = 3
  else
    nleaves = 2
  end if
  nleavesalloc = nleaves*stride
  if (stride > 1) then
    do i = 1, nleaves
      mine(i) = stride*(i - 1)
    end do
  end if

! Left periodic neighbor
  remote(1)%rank = modulo(rank + size - 1, size)
  remote(1)%index = 1*stride
! Right periodic neighbor
  remote(2)%rank = modulo(rank + 1, size)
  remote(2)%index = 0*stride
  if (rank > 0) then !               All processes reference rank 0, index
    remote(3)%rank = 0
    remote(3)%index = 2*stride
  end if

!  Create a star forest for communication
  PetscCallA(PetscSFCreate(PETSC_COMM_WORLD, sf, ierr))
  PetscCallA(PetscSFSetFromOptions(sf, ierr))
  PetscCallA(PetscSFSetGraph(sf, nrootsalloc, nleaves, mine, PETSC_COPY_VALUES, remote, PETSC_COPY_VALUES, ierr))
  PetscCallA(PetscSFSetUp(sf, ierr))

!   View graph, mostly useful for debugging purposes.
  PetscCallA(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL, ierr))
  PetscCallA(PetscSFView(sf, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD, ierr))

!   Allocate space for send and receive buffers. This example communicates PetscInt, but other types, including
!     * user-defined structures, could also be used.
!     Set rootdata buffer to be broadcast
  do i = 1, nrootsalloc
    rootdata(i) = -1
  end do
  do i = 1, nroots
    rootdata(1 + (i - 1)*stride) = 100*(rank + 1) + i - 1
  end do

!     Initialize local buffer, these values are never used.
  do i = 1, nleavesalloc
    leafdata(i) = -1
  end do

!     Broadcast entries from rootdata to leafdata. Computation or other communication can be performed between the begin and end calls.
  PetscCallA(PetscSFBcastBegin(sf, MPIU_INTEGER, rootdata, leafdata, MPI_REPLACE, ierr))
  PetscCallA(PetscSFBcastEnd(sf, MPIU_INTEGER, rootdata, leafdata, MPI_REPLACE, ierr))
  PetscCallA(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, '## Bcast Rootdata\n', ierr))
  PetscCallA(PetscIntView(nrootsalloc, rootdata, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, '## Bcast Leafdata\n', ierr))
  PetscCallA(PetscIntView(nleavesalloc, leafdata, PETSC_VIEWER_STDOUT_WORLD, ierr))

!     Reduce entries from leafdata to rootdata. Computation or other communication can be performed between the begin and end calls.
  PetscCallA(PetscSFReduceBegin(sf, MPIU_INTEGER, leafdata, rootdata, MPI_SUM, ierr))
  PetscCallA(PetscSFReduceEnd(sf, MPIU_INTEGER, leafdata, rootdata, MPI_SUM, ierr))
  PetscCallA(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, '## Reduce Leafdata\n', ierr))
  PetscCallA(PetscIntView(nleavesalloc, leafdata, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD, '## Reduce Rootdata\n', ierr))
  PetscCallA(PetscIntView(nrootsalloc, rootdata, PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(PetscSFGetGraph(sf, gnroots, gnleaves, gmine, gremote, ierr))
  PetscCheckA(gnleaves == nleaves, PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'nleaves returned from PetscSFGetGraph() does not match that set with PetscSFSetGraph()')
  do i = 1, nleaves
    PetscCheckA(gmine(i) == mine(i), PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'Root from PetscSFGetGraph() does not match that set with PetscSFSetGraph()')
  end do
  do i = 1, nleaves
    PetscCheckA(gremote(i)%index == remote(i)%index, PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'Leaf from PetscSFGetGraph() does not match that set with PetscSFSetGraph()')
  end do
  PetscCallA(PetscSFRestoreGraph(sf, gnroots, gnleaves, gmine, gremote, ierr))

! Test PetscSFGet{Leaf,Root}Ranks
  PetscCallA(PetscSFGetLeafRanks(sf, niranks, iranks, ioffset, irootloc, ierr))
  PetscCallA(PetscSFGetRootRanks(sf, nranks, ranks, roffset, rmine, rremote, ierr))

!    Clean storage for star forest.
  PetscCallA(PetscSFDestroy(sf, ierr))

!  Create a star forest with continuous leaves and hence no buffer
  PetscCallA(PetscSFCreate(PETSC_COMM_WORLD, sf, ierr))
  PetscCallA(PetscSFSetFromOptions(sf, ierr))
  PetscCallA(PetscSFSetGraph(sf, nrootsalloc, nleaves, PETSC_NULL_INTEGER_ARRAY, PETSC_COPY_VALUES, remote, PETSC_COPY_VALUES, ierr))
  PetscCallA(PetscSFSetUp(sf, ierr))

!   View graph, mostly useful for debugging purposes.
  PetscCallA(PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD, PETSC_VIEWER_ASCII_INFO_DETAIL, ierr))
  PetscCallA(PetscSFView(sf, PETSC_VIEWER_STDOUT_WORLD, ierr))
  PetscCallA(PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD, ierr))

  PetscCallA(PetscSFGetGraph(sf, gnroots, gnleaves, gmine, gremote, ierr))
  PetscCheckA(loc(gmine) == loc(PETSC_NULL_INTEGER), PETSC_COMM_WORLD, PETSC_ERR_PLIB, 'Leaves from PetscSFGetGraph() not null as expected')
  PetscCallA(PetscSFRestoreGraph(sf, gnroots, gnleaves, gmine, gremote, ierr))
  PetscCallA(PetscSFDestroy(sf, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!  build:
!    requires: defined(PETSC_HAVE_FORTRAN_TYPE_STAR)
!
!  test:
!    nsize: 3
!
!TEST*/
