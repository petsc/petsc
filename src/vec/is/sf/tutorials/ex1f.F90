
!    Description: A star forest is a simple tree with one root and zero or more leaves.
!    Many common communication patterns can be expressed as updates of rootdata using leafdata and vice-versa.
!     This example creates a star forest, communicates values using the graph  views the graph, then destroys it.
!
!     This is a copy of ex1.c but currently only tests the broadcast operation

      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

      PetscErrorCode                ierr
      PetscInt                      i,nroots,nrootsalloc,nleaves,nleavesalloc,mine(6),stride
      type(PetscSFNode)             remote(6)
      PetscMPIInt                   rank,size
      PetscSF                       sf
      PetscInt                      rootdata(6),leafdata(6)

! used with PetscSFGetGraph()
      type(PetscSFNode), pointer :: gremote(:)
      PetscInt, pointer ::          gmine(:)
      PetscInt                      gnroots,gnleaves;

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      stride = 2
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);CHKERRA(ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr);CHKERRA(ierr)

      if (rank == 0) then
         nroots = 3
      else
         nroots = 2
      endif
      nrootsalloc  = nroots * stride;
      if (rank > 0) then
         nleaves = 3
      else
         nleaves = 2
      endif
      nleavesalloc = nleaves * stride
      if (stride > 1) then
         do i=1,nleaves
            mine(i) = stride * (i-1)
         enddo
      endif

! Left periodic neighbor
      remote(1)%rank  = modulo(rank+size-1,size)
      remote(1)%index = 1 * stride
! Right periodic neighbor
      remote(2)%rank  = modulo(rank+1,size)
      remote(2)%index = 0 * stride
      if (rank > 0) then !               All processes reference rank 0, index
         remote(3)%rank  = 0
         remote(3)%index = 2 * stride
      endif

!  Create a star forest for communication
      call PetscSFCreate(PETSC_COMM_WORLD,sf,ierr);CHKERRA(ierr)
      call PetscSFSetFromOptions(sf,ierr);CHKERRA(ierr)
      call PetscSFSetGraph(sf,nrootsalloc,nleaves,mine,PETSC_COPY_VALUES,remote,PETSC_COPY_VALUES,ierr);CHKERRA(ierr)
      call PetscSFSetUp(sf,ierr);CHKERRA(ierr)

!   View graph, mostly useful for debugging purposes.
      call PetscViewerPushFormat(PETSC_VIEWER_STDOUT_WORLD,PETSC_VIEWER_ASCII_INFO_DETAIL,ierr);CHKERRA(ierr)
      call PetscSFView(sf,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
      call PetscViewerPopFormat(PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

!   Allocate space for send and receive buffers. This example communicates PetscInt, but other types, including
!     * user-defined structures, could also be used.
!     Set rootdata buffer to be broadcast
      do i=1,nrootsalloc
         rootdata(i) = -1
      enddo
      do i=1,nroots
         rootdata(1 + (i-1)*stride) = 100*(rank+1) + i - 1
      enddo

!     Initialize local buffer, these values are never used.
      do i=1,nleavesalloc
         leafdata(i) = -1
      enddo

!     Broadcast entries from rootdata to leafdata. Computation or other communication can be performed between the begin and end calls.
      call PetscSFBcastBegin(sf,MPIU_INTEGER,rootdata,leafdata,MPI_REPLACE,ierr);CHKERRA(ierr)
      call PetscSFBcastEnd(sf,MPIU_INTEGER,rootdata,leafdata,MPI_REPLACE,ierr);CHKERRA(ierr)
      call PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Rootdata\n",ierr);CHKERRA(ierr)
      call PetscIntView(nrootsalloc,rootdata,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)
      call PetscViewerASCIIPrintf(PETSC_VIEWER_STDOUT_WORLD,"## Bcast Leafdata\n",ierr);CHKERRA(ierr)
      call PetscIntView(nleavesalloc,leafdata,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

      call PetscSFGetGraph(sf,gnroots,gnleaves,gmine,gremote,ierr);CHKERRA(ierr)
      if (gnleaves .ne. nleaves) then; SETERRA(PETSC_COMM_WORLD,PETSC_ERR_PLIB,'nleaves returned from PetscSFGetGraph() does not match that set with PetscSFSetGraph()'); endif
      do i=1,nleaves
         if (gmine(i) .ne. mine(i)) then; SETERRA(PETSC_COMM_WORLD,PETSC_ERR_PLIB,'Root from PetscSFGetGraph() does not match that set with PetscSFSetGraph()'); endif
      enddo
      do i=1,nleaves
         if (gremote(i)%index .ne. remote(i)%index) then; SETERRA(PETSC_COMM_WORLD,PETSC_ERR_PLIB,'Leaf from PetscSFGetGraph() does not match that set with PetscSFSetGraph()'); endif
      enddo

      deallocate(gremote)
!    Clean storage for star forest.
      call PetscSFDestroy(sf,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr);
  end

!/*TEST
!  build:
!    requires: define(PETSC_HAVE_FORTRAN_TYPE_STAR)
!
!  test:
!    nsize: 3
!
!TEST*/
