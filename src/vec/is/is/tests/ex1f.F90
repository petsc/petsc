!
!
!  Formatted test for IS general routines
!
      program main
#include <petsc/finclude/petscis.h>
      use petscis
      implicit none

       PetscErrorCode ierr
       PetscInt i,n,indices(1004),ii(1)
       PetscMPIInt size,rank
       PetscOffset iis
       IS          is,newis
       PetscBool   flag,compute,permanent

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
       if (ierr .ne. 0) then
         print*,'Unable to initialize PETSc'
         stop
       endif
       call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
       call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)

!     Test IS of size 0

       n = 0
       call ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,is,ierr);CHKERRA(ierr)
       call ISGetLocalSize(is,n,ierr);CHKERRA(ierr)
       if (n .ne. 0) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error getting size of zero IS'); endif
       call ISDestroy(is,ierr)


!     Create large IS and test ISGetIndices(,ierr)
!     fortran indices start from 1 - but IS indices start from 0
      n = 1000 + rank
      do 10, i=1,n
        indices(i) = rank + i-1
 10   continue
      call ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,is,ierr);CHKERRA(ierr)
      call ISGetIndices(is,ii,iis,ierr);CHKERRA(ierr)
      do 20, i=1,n
        if (ii(i+iis) .ne. indices(i)) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error getting indices'); endif
 20   continue
      call ISRestoreIndices(is,ii,iis,ierr);CHKERRA(ierr)

!     Check identity and permutation

      compute = PETSC_TRUE
      permanent = PETSC_FALSE
      call ISPermutation(is,flag,ierr);CHKERRA(ierr)
      if (flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking permutation'); endif
      call ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,compute,flag,ierr);CHKERRA(ierr)
      !if ((rank .eq. 0) .and. (.not. flag)) SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_PERMUTATION,IS_LOCAL)")
      !if (rank .eq. 0 .and. flag) SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_PERMUTATION,IS_LOCAL)")
      call ISIdentity(is,flag,ierr);CHKERRA(ierr)
      if ((rank .eq. 0) .and. (.not. flag)) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking identity'); endif
      if ((rank .ne. 0) .and. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking identity'); endif
      call ISSetInfo(is,IS_PERMUTATION,IS_LOCAL,permanent,PETSC_TRUE,ierr);CHKERRA(ierr)
      call ISSetInfo(is,IS_IDENTITY,IS_LOCAL,permanent,PETSC_TRUE,ierr);CHKERRA(ierr)
      call ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,compute,flag,ierr);CHKERRA(ierr)
      if (.not. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking permutation second time'); endif
      call ISGetInfo(is,IS_IDENTITY,IS_LOCAL,compute,flag,ierr);CHKERRA(ierr)
      if (.not. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking identity second time'); endif
      call ISClearInfoCache(is,PETSC_TRUE,ierr);CHKERRA(ierr)

!     Check equality of index sets

      call ISEqual(is,is,flag,ierr);CHKERRA(ierr)
      if (.not. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking equal'); endif

!     Sorting

      call ISSort(is,ierr);CHKERRA(ierr)
      call ISSorted(is,flag,ierr);CHKERRA(ierr)
      if (.not. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking sorted'); endif

!     Thinks it is a different type?

      call PetscObjectTypeCompare(is,ISSTRIDE,flag,ierr);CHKERRA(ierr)
      if (flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking stride'); endif
      call PetscObjectTypeCompare(is,ISBLOCK,flag,ierr);CHKERRA(ierr)
      if (flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking block'); endif

      call ISDestroy(is,ierr);CHKERRA(ierr)

!     Inverting permutation

      do 30, i=1,n
        indices(i) = n - i
 30   continue

      call ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,is,ierr);CHKERRA(ierr)
      call ISSetPermutation(is,ierr);CHKERRA(ierr)
      call ISInvertPermutation(is,PETSC_DECIDE,newis,ierr);CHKERRA(ierr)
      call ISGetIndices(newis,ii,iis,ierr);CHKERRA(ierr)
      do 40, i=1,n
        if (ii(iis+i) .ne. n - i) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error getting permutation indices'); endif
 40   continue
      call ISRestoreIndices(newis,ii,iis,ierr);CHKERRA(ierr)
      call ISDestroy(newis,ierr);CHKERRA(ierr)
      call ISDestroy(is,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!     nsize: {{1 2 3 4 5}}
!     output_file: output/ex1_1.out
!
!TEST*/
