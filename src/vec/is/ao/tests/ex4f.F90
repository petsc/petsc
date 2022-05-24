!
!    Test AO with on IS with 0 entries - Fortran version of ex4.c
!
      program main
#include <petsc/finclude/petscao.h>
      use petscao
      implicit none

      PetscErrorCode ierr
      AO             ao
      PetscInt       localvert(4),nlocal
      PetscMPIInt    rank
      IS             is
      PetscInt       one,zero

!  Needed to work with 64 bit integers from Fortran
      one  = 1
      zero = 0

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

      nlocal = 0
      if (rank .eq. 0) then
         nlocal = 4
         localvert(1) = 0
         localvert(2) = 1
         localvert(3) = 2
         localvert(4) = 3
      endif

!     Test AOCreateBasic()
      PetscCallA(AOCreateBasic(PETSC_COMM_WORLD, nlocal, localvert,PETSC_NULL_INTEGER,ao,ierr))
      PetscCallA(AODestroy(ao,ierr))

!     Test AOCreateMemoryScalable()
      PetscCallA(AOCreateMemoryScalable(PETSC_COMM_WORLD, nlocal, localvert,PETSC_NULL_INTEGER,ao,ierr))
      PetscCallA(AODestroy(ao,ierr))

      PetscCallA(AOCreate(PETSC_COMM_WORLD,ao,ierr))
      PetscCallA(ISCreateStride(PETSC_COMM_WORLD,one,zero,one,is,ierr))
      PetscCallA(AOSetIS(ao,is,is,ierr))
      PetscCallA(AOSetType(ao,AOMEMORYSCALABLE,ierr))
      PetscCallA(ISDestroy(is,ierr))
      PetscCallA(AODestroy(ao,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!     output_file: output/ex4_1.out
!
!   test:
!      suffix: 2
!      nsize: 2
!      output_file: output/ex4_1.out
!
!TEST*/
