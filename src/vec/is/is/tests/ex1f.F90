!
!
!  Formatted test for IS general routines
!
      program main
#include <petsc/finclude/petscis.h>
      use petscis
      implicit none

       PetscErrorCode ierr
       PetscInt i,n,indices(1004)
       PetscInt, pointer :: ii(:)
       PetscMPIInt size,rank
       IS          is,newis
       PetscBool   flag,compute,permanent

       PetscCallA(PetscInitialize(ierr))
       PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
       PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,size,ierr))

!     Test IS of size 0

       n = 0
       PetscCallA(ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,is,ierr))
       PetscCallA(ISGetLocalSize(is,n,ierr))
       if (n .ne. 0) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error getting size of zero IS'); endif
       PetscCallA(ISDestroy(is,ierr))

!     Create large IS and test ISGetIndices(,ierr))
!     fortran indices start from 1 - but IS indices start from 0
      n = 1000 + rank
      do 10, i=1,n
        indices(i) = rank + i-1
 10   continue
      PetscCallA(ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,is,ierr))
      PetscCallA(ISGetIndicesF90(is,ii,ierr))
      do 20, i=1,n
        if (ii(i) .ne. indices(i)) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error getting indices'); endif
 20   continue
      PetscCallA(ISRestoreIndicesF90(is,ii,ierr))

!     Check identity and permutation

      compute = PETSC_TRUE
      permanent = PETSC_FALSE
      PetscCallA(ISPermutation(is,flag,ierr))
      if (flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking permutation'); endif
      PetscCallA(ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,compute,flag,ierr))
      !if ((rank .eq. 0) .and. (.not. flag)) SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_PERMUTATION,IS_LOCAL)")
      !if (rank .eq. 0 .and. flag) SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ISGetInfo(IS_PERMUTATION,IS_LOCAL)")
      PetscCallA(ISIdentity(is,flag,ierr))
      if ((rank .eq. 0) .and. (.not. flag)) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking identity'); endif
      if ((rank .ne. 0) .and. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking identity'); endif
      PetscCallA(ISSetInfo(is,IS_PERMUTATION,IS_LOCAL,permanent,PETSC_TRUE,ierr))
      PetscCallA(ISSetInfo(is,IS_IDENTITY,IS_LOCAL,permanent,PETSC_TRUE,ierr))
      PetscCallA(ISGetInfo(is,IS_PERMUTATION,IS_LOCAL,compute,flag,ierr))
      if (.not. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking permutation second time'); endif
      PetscCallA(ISGetInfo(is,IS_IDENTITY,IS_LOCAL,compute,flag,ierr))
      if (.not. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking identity second time'); endif
      PetscCallA(ISClearInfoCache(is,PETSC_TRUE,ierr))

!     Check equality of index sets

      PetscCallA(ISEqual(is,is,flag,ierr))
      if (.not. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking equal'); endif

!     Sorting

      PetscCallA(ISSort(is,ierr))
      PetscCallA(ISSorted(is,flag,ierr))
      if (.not. flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking sorted'); endif

!     Thinks it is a different type?

      PetscCallA(PetscObjectTypeCompare(is,ISSTRIDE,flag,ierr))
      if (flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking stride'); endif
      PetscCallA(PetscObjectTypeCompare(is,ISBLOCK,flag,ierr))
      if (flag) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error checking block'); endif

      PetscCallA(ISDestroy(is,ierr))

!     Inverting permutation

      do 30, i=1,n
        indices(i) = n - i
 30   continue

      PetscCallA(ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,is,ierr))
      PetscCallA(ISSetPermutation(is,ierr))
      PetscCallA(ISInvertPermutation(is,PETSC_DECIDE,newis,ierr))
      PetscCallA(ISGetIndicesF90(newis,ii,ierr))
      do 40, i=1,n
        if (ii(i) .ne. n - i) then; SETERRA(PETSC_COMM_SELF,PETSC_ERR_PLIB,'Error getting permutation indices'); endif
 40   continue
      PetscCallA(ISRestoreIndicesF90(newis,ii,ierr))
      PetscCallA(ISDestroy(newis,ierr))
      PetscCallA(ISDestroy(is,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!     nsize: {{1 2 3 4 5}}
!     output_file: output/ex1_1.out
!
!TEST*/
