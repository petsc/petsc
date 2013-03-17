!
!
!  Formatted test for IS general routines
!
      program main
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscis.h>


       PetscErrorCode ierr
       PetscInt i,n,indices(1000),ii(1)
       PetscMPIInt size,rank
       PetscOffset iis
       IS          is,newis
       PetscBool   flag

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
           CHKERRQ(ierr)
       call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
       call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)

!     Test IS of size 0

       n = 0
       call ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,    &
     &                      is,ierr)
           CHKERRQ(ierr)
       call ISGetLocalSize(is,n,ierr)
           CHKERRQ(ierr)
       if (n .ne. 0) then
         print*, 'Error getting size of zero IS'
         stop
       endif
       call ISDestroy(is,ierr)


!     Create large IS and test ISGetIndices(,ierr)
!     fortran indices start from 1 - but IS indices start from 0
      n = 1000
      do 10, i=1,n
        indices(i) = i-1
 10   continue
      call ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,       &
     &                     is,ierr)
           CHKERRQ(ierr)
      call ISGetIndices(is,ii,iis,ierr)
           CHKERRQ(ierr)
      do 20, i=1,n
        if (ii(i+iis) .ne. indices(i)) then
           print*, 'Error getting indices'
           stop
        endif
 20   continue
      call ISRestoreIndices(is,ii,iis,ierr)
           CHKERRQ(ierr)

!     Check identity and permutation

      call ISPermutation(is,flag,ierr)
           CHKERRQ(ierr)
      if (flag) then
         print*, 'Error checking permutation'
         stop
      endif
      call ISIdentity(is,flag,ierr)
           CHKERRQ(ierr)
      if (.not. flag) then
         print*, 'Error checking identity'
         stop
      endif
      call ISSetPermutation(is,ierr)
           CHKERRQ(ierr)
      call ISSetIdentity(is,ierr)
           CHKERRQ(ierr)
      call ISPermutation(is,flag,ierr)
           CHKERRQ(ierr)
      if (.not. flag) then
         print*, 'Error checking permutation second time'
         stop
      endif
      call ISIdentity(is,flag,ierr)
           CHKERRQ(ierr)
      if (.not. flag) then
         print*, 'Error checking identity second time'
         stop
      endif

!     Check equality of index sets

      call ISEqual(is,is,flag,ierr)
           CHKERRQ(ierr)
      if (.not. flag) then
         print*, 'Error checking equal'
         stop
      endif

!     Sorting

      call ISSort(is,ierr)
           CHKERRQ(ierr)
      call ISSorted(is,flag,ierr)
           CHKERRQ(ierr)
      if (.not. flag) then
         print*, 'Error checking sorted'
         stop
      endif

!     Thinks it is a different type?

      call PetscObjectTypeCompare(is,ISSTRIDE,flag,ierr)
           CHKERRQ(ierr)
      if (flag) then
         print*, 'Error checking stride'
         stop
      endif
      call PetscObjectTypeCompare(is,ISBLOCK,flag,ierr)
           CHKERRQ(ierr)
      if (flag) then
         print*, 'Error checking block'
         stop
      endif

      call ISDestroy(is,ierr)
           CHKERRQ(ierr)

!     Inverting permutation

      do 30, i=1,n
        indices(i) = n - i
 30   continue

      call ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,         &
     &                     is,ierr)
           CHKERRQ(ierr)
      call ISSetPermutation(is,ierr)
           CHKERRQ(ierr)
      call ISInvertPermutation(is,PETSC_DECIDE,newis,ierr)
           CHKERRQ(ierr)
      call ISGetIndices(newis,ii,iis,ierr)
           CHKERRQ(ierr)
      do 40, i=1,n
        if (ii(iis+i) .ne. n - i) then
          print*, 'Error getting permutation indices'
          stop
       endif
 40   continue
      call ISRestoreIndices(newis,ii,iis,ierr)
           CHKERRQ(ierr)
      call ISDestroy(newis,ierr)
           CHKERRQ(ierr)
      call ISDestroy(is,ierr)
           CHKERRQ(ierr)
      call PetscFinalize(ierr)
      end







