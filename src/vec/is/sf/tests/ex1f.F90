!
!  Tests VecScatterCreateToAll Fortran stub
      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

      PetscErrorCode ierr
      PetscInt  nlocal, row
      PetscScalar num
      PetscMPIInt rank
      Vec v1, v2
      VecScatter toall

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

      nlocal = 1
      call VecCreateMPI(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,v1,ierr)

      row = rank
      num = rank
      call VecSetValue(v1,row,num,INSERT_VALUES,ierr)
      call VecAssemblyBegin(v1,ierr)
      call VecAssemblyEnd(v1,ierr)

      call VecScatterCreateToAll(v1,toall,v2,ierr)

      call VecScatterBegin(toall,v1,v2,INSERT_VALUES,SCATTER_FORWARD,ierr)
      call VecScatterEnd(toall,v1,v2,INSERT_VALUES,SCATTER_FORWARD,ierr)

      if (rank.eq.2) then
         call PetscObjectSetName(v2, 'v2',ierr)
         call VecView(v2,PETSC_VIEWER_STDOUT_SELF,ierr)
      end if

      call VecScatterDestroy(toall,ierr)
      call VecDestroy(v1,ierr)
      call VecDestroy(v2,ierr)

      call PetscFinalize(ierr)
      end

!/*TEST
!
!     test:
!       nsize: 4
!
!TEST*/
