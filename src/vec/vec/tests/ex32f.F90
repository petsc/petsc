!
!
!  Tests PescOffsetFortran()
!  duplicated
      program main
#include <petsc/finclude/petscvec.h>
      use petscmpi  ! or mpi or mpi_f08
      use petscvec
       implicit none

      PetscErrorCode ierr
      PetscInt  n
      PetscMPIInt size,zero

      PetscScalar  v_v1(1),v_v2(1)
      Vec     v
      PetscInt i
      PetscOffset i_v1,i_v2

      zero=0
      n=8
      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))
      if (size .gt. 1) then
        print*,'Example for one processor only'
        PetscCallMPIA(MPI_Abort(MPI_COMM_WORLD,zero,ierr))
      endif

      PetscCallA(VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,n,v,ierr))
      PetscCallA(VecGetArray(v,v_v1,i_v1,ierr))

      do 10, i=1,n
        v_v1(i_v1 + i) = i
 10   continue
      PetscCallA(VecRestoreArray(v,v_v1,i_v1,ierr))

      PetscCallA(VecView(v,PETSC_VIEWER_STDOUT_WORLD,ierr))

      PetscCallA(VecGetArray(v,v_v1,i_v1,ierr))
      PetscCallA(PetscOffsetFortran(v_v2,v_v1,i_v2,ierr))
      i_v2 = i_v1 + i_v2
      do 20, i=1,n
        print*,i,v_v2(i_v2 + i)
 20   continue
      PetscCallA(VecRestoreArray(v,v_v1,i_v1,ierr))

      PetscCallA(VecDestroy(v,ierr))
      PetscCallA(PetscFinalize(ierr))

      end

!/*TEST
!
!     test:
!       requires: !complex
!
!TEST*/

