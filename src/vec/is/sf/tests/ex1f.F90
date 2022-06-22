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

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

      nlocal = 1
      PetscCallA(VecCreateMPI(PETSC_COMM_WORLD,nlocal,PETSC_DECIDE,v1,ierr))

      row = rank
      num = rank
      PetscCallA(VecSetValue(v1,row,num,INSERT_VALUES,ierr))
      PetscCallA(VecAssemblyBegin(v1,ierr))
      PetscCallA(VecAssemblyEnd(v1,ierr))

      PetscCallA(VecScatterCreateToAll(v1,toall,v2,ierr))

      PetscCallA(VecScatterBegin(toall,v1,v2,INSERT_VALUES,SCATTER_FORWARD,ierr))
      PetscCallA(VecScatterEnd(toall,v1,v2,INSERT_VALUES,SCATTER_FORWARD,ierr))

      PetscCallA(VecScatterDestroy(toall,ierr))
! Destroy v2 and then re-create it in VecScatterCreateToAll() to test if petsc can differentiate NULL projects with destroyed objects
      PetscCallA(VecDestroy(v2,ierr))

      PetscCallA(VecScatterCreateToAll(v1,toall,v2,ierr))
      PetscCallA(VecScatterBegin(toall,v1,v2,INSERT_VALUES,SCATTER_FORWARD,ierr))
      PetscCallA(VecScatterEnd(toall,v1,v2,INSERT_VALUES,SCATTER_FORWARD,ierr))

      if (rank.eq.2) then
         PetscCallA(PetscObjectSetName(v2, 'v2',ierr))
         PetscCallA(VecView(v2,PETSC_VIEWER_STDOUT_SELF,ierr))
      end if

      PetscCallA(VecScatterDestroy(toall,ierr))
      PetscCallA(VecDestroy(v1,ierr))
      PetscCallA(VecDestroy(v2,ierr))
! It is OK to destroy again
      PetscCallA(VecDestroy(v2,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!     test:
!       nsize: 4
!
!TEST*/
