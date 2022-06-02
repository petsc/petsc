!
!
!     Test for PetscFOpen() from Fortran
!
      program main
#include <petsc/finclude/petscsys.h>
      use petscsys
      implicit none

      PetscErrorCode ierr
      PetscMPIInt rank
      PetscFortranAddr file
      character*100    joe

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

      PetscCallA(PetscFOpen(PETSC_COMM_WORLD,'testfile','w',file,ierr))

      PetscCallA(PetscFPrintf(PETSC_COMM_WORLD,file,'Hi once \n',ierr))
      PetscCallA(PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file,'Hi twice \n',ierr))
      PetscCallA(PetscSynchronizedFlush(PETSC_COMM_WORLD,file,ierr))

      write (FMT=*,UNIT=joe) 'greetings from ',rank,'\n'
      PetscCallA(PetscSynchronizedFPrintf(PETSC_COMM_WORLD,file,joe,ierr))
      PetscCallA(PetscSynchronizedFlush(PETSC_COMM_WORLD,file,ierr))

      PetscCallA(PetscFClose(PETSC_COMM_WORLD,file,ierr))

      PetscCallA(PetscSynchronizedPrintf(PETSC_COMM_WORLD,'Hi twice \n',ierr))
      PetscCallA(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT,ierr))

      PetscCallA(PetscFinalize(ierr))
      end

!
!/*TEST
!
!   test:
!      nsize: 3
!
!TEST*/
