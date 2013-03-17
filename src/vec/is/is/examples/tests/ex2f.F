!
!  Formatted Test for IS stride routines
!
      program main
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscis.h>

      PetscErrorCode ierr
      PetscInt  i,n,ii(1),start
      PetscInt  stride,ssize,first
      IS          is
      PetscBool   flag
      PetscOffset iis

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

!     Test IS of size 0
      ssize = 0
      stride = 0
      first = 2
      call ISCreateStride(PETSC_COMM_SELF,ssize,stride,first,is,ierr)
      call ISGetLocalSize(is,n,ierr)
      if (n .ne. 0) then
        SETERRQ(PETSC_COMM_SELF,1,0,ierr)
      endif
      call ISStrideGetInfo(is,start,stride,ierr)
      if (start .ne. 0) then
         SETERRQ(PETSC_COMM_SELF,1,0,ierr)
      endif
      if (stride .ne. 2) then
        SETERRQ(PETSC_COMM_SELF,1,0,ierr)
      endif
      call PetscObjectTypeCompare(is,ISSTRIDE,flag,ierr)
      if (.not. flag) then
        SETERRQ(PETSC_COMM_SELF,1,0,ierr)
      endif
      call ISGetIndices(is,ii,iis,ierr)
      call ISRestoreIndices(is,ii,iis,ierr)
      call ISDestroy(is,ierr)

!     Test ISGetIndices()

      ssize = 10000
      stride = -8
      first = 3
      call ISCreateStride(PETSC_COMM_SELF,ssize,stride,first,is,ierr)
      call ISGetLocalSize(is,n,ierr)
      call ISGetIndices(is,ii,iis,ierr)
      do 10, i=1,10000
        if (ii(i+iis) .ne. -11 + 3*i) then
          SETERRQ(PETSC_COMM_SELF,1,0,ierr)
        endif
 10   continue
      call ISRestoreIndices(is,ii,iis,ierr)
      call ISDestroy(is,ierr)

      call PetscFinalize(ierr)
      end







