!
!
! Description: Illustrates the use of VecCreateGhost()
!
!
!      Ghost padding is one way to handle local calculations that
!      involve values from other processors. VecCreateGhost() provides
!      a way to create vectors with extra room at the end of the vector
!      array to contain the needed ghost values from other processors,
!      vector computations are otherwise unaffected.
!

      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

      PetscMPIInt rank,mySize
      PetscInt nlocal,nghost,ifrom(2)
      PetscErrorCode ierr
      PetscInt i,rstart,rend,ione
      PetscBool   flag
      PetscScalar  value,tarray(20)
      Vec          lx,gx,gxs
      PetscViewer  subviewer

      nlocal = 6
      nghost = 2

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr /= 0) then
        print*,'PetscInitialize failed'
        stop
      endif
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
      call MPI_Comm_size(PETSC_COMM_WORLD,mySize,ierr)

      if (mySize /= 2) then; SETERRA(PETSC_COMM_WORLD,PETSC_ERR_WRONG_MPI_SIZE,'Requires 2 processors'); endif

!
!     Construct a two dimensional graph connecting nlocal degrees of
!     freedom per processor. From this we will generate the global
!     indices of needed ghost values
!
!     For simplicity we generate the entire graph on each processor:
!     in real application the graph would stored in parallel, but this
!     example is only to demonstrate the management of ghost padding
!     with VecCreateGhost().
!
!     In this example we consider the vector as representing
!     degrees of freedom in a one dimensional grid with periodic
!     boundary conditions.
!
!        ----Processor  1---------  ----Processor 2 --------
!         0    1   2   3   4    5    6    7   8   9   10   11
!                               |----|
!         |-------------------------------------------------|
!

      if (rank .eq. 0) then
        ifrom(1) = 11
        ifrom(2) = 6
      else
        ifrom(1) = 0
        ifrom(2) = 5
      endif

!     Create the vector with two slots for ghost points. Note that both
!     the local vector (lx) and the global vector (gx) share the same
!     array for storing vector values.

      call PetscOptionsHasName(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,   &
     &                         '-allocate',flag,ierr)
      if (flag) then
        call VecCreateGhostWithArray(PETSC_COMM_WORLD,nlocal,            &
     &        PETSC_DECIDE,nghost,ifrom,tarray,gxs,ierr)
      else
        call VecCreateGhost(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,        &
     &       nghost,ifrom,gxs,ierr)
      endif

!      Test VecDuplicate

       call VecDuplicate(gxs,gx,ierr)
       call VecDestroy(gxs,ierr)

!      Access the local Form

       call VecGhostGetLocalForm(gx,lx,ierr)

!     Set the values from 0 to 12 into the 'global' vector

       call VecGetOwnershipRange(gx,rstart,rend,ierr)

       ione = 1
       do 10, i=rstart,rend-1
         value = real(i)
         call VecSetValues(gx,ione,i,value,INSERT_VALUES,ierr)
 10    continue

       call VecAssemblyBegin(gx,ierr)
       call VecAssemblyEnd(gx,ierr)

       call VecGhostUpdateBegin(gx,INSERT_VALUES,SCATTER_FORWARD,ierr)
       call VecGhostUpdateEnd(gx,INSERT_VALUES,SCATTER_FORWARD,ierr)

!     Print out each vector, including the ghost padding region.

       call PetscViewerGetSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,subviewer,ierr)
       call VecView(lx,subviewer,ierr)
       call PetscViewerRestoreSubViewer(PETSC_VIEWER_STDOUT_WORLD,PETSC_COMM_SELF,subviewer,ierr)

       call VecGhostRestoreLocalForm(gx,lx,ierr)
       call VecDestroy(gx,ierr)
       call PetscFinalize(ierr)
       end

!/*TEST
!
!     test:
!       nsize: 2
!
!     test:
!       suffix: 2
!       nsize: 2
!       args: -allocate
!
!TEST*/

