!
!  "$Id: ex1f.F,v 1.15 2000/02/02 20:08:36 bsmith Exp balay $"
!
!  Formatted test for IS general routines
!
      implicit none
#include "finclude/petsc.h"
#include "finclude/petscis.h"


       integer     i,n,ierr,indices(1000),rank,size,ii(1)
       PetscOffset iis
       IS          is,newis
       PetscTruth  flag

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
           CHKERRA(ierr)
       call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)
       call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)

!     Test IS of size 0 

       call ISCreateGeneral(PETSC_COMM_SELF,0,indices,is,ierr) 
           CHKERRA(ierr)
       call ISGetSize(is,n,ierr) 
           CHKERRA(ierr)
       if (n .ne. 0) then
         print*, 'Error getting size of zero IS'
         stop
       endif
       call ISDestroy(is,ierr) 


!     Create large IS and test ISGetIndices(,ierr)

      n = 1000
      do 10, i=1,n
        indices(i) = rank + i
 10   continue
      call ISCreateGeneral(PETSC_COMM_SELF,n,indices,is,ierr) 
           CHKERRA(ierr)
      call ISGetIndices(is,ii,iis,ierr) 
           CHKERRA(ierr)
      do 20, i=1,n
        if (ii(i+iis) .ne. indices(i)) then
           print*, 'Error getting indices'
           stop
        endif
 20   continue
      call ISRestoreIndices(is,ii,iis,ierr) 
           CHKERRA(ierr)

!     Check identity and permutation 
  
      call ISPermutation(is,flag,ierr) 
           CHKERRA(ierr)
      if (flag .eq. PETSC_TRUE) then
         print*, 'Error checking permutation'
         stop
      endif
      call ISIdentity(is,flag,ierr) 
           CHKERRA(ierr)
      if (flag .eq. PETSC_TRUE) then
         print*, 'Error checking identity'
         stop
      endif
      call ISSetPermutation(is,ierr) 
           CHKERRA(ierr)
      call ISSetIdentity(is,ierr)  
           CHKERRA(ierr)
      call ISPermutation(is,flag,ierr) 
           CHKERRA(ierr)
      if (flag .ne. PETSC_TRUE) then
         print*, 'Error checking permutation second time'
         stop
      endif
      call ISIdentity(is,flag,ierr) 
           CHKERRA(ierr)
      if (flag .ne. PETSC_TRUE) then
         print*, 'Error checking identity second time'
         stop
      endif

!     Check equality of index sets 

      call ISEqual(is,is,flag,ierr) 
           CHKERRA(ierr)
      if (flag .ne. PETSC_TRUE) then
         print*, 'Error checking equal'
         stop
      endif

!     Sorting 

      call ISSort(is,ierr) 
           CHKERRA(ierr)
      call ISSorted(is,flag,ierr) 
           CHKERRA(ierr)
      if (flag .ne. PETSC_TRUE) then
         print*, 'Error checking sorted'
         stop
      endif

!     Thinks it is a different type?

      call ISStride(is,flag,ierr) 
           CHKERRA(ierr)
      if (flag .eq. PETSC_TRUE) then
         print*, 'Error checking stride'
         stop
      endif
      call ISBlock(is,flag,ierr) 
           CHKERRA(ierr)
      if (flag .eq. PETSC_TRUE) then
         print*, 'Error checking block'
         stop
      endif

      call ISDestroy(is,ierr) 
           CHKERRA(ierr)

!     Inverting permutation

      do 30, i=1,n
        indices(i) = n - i
 30   continue

      call ISCreateGeneral(PETSC_COMM_SELF,n,indices,is,ierr) 
           CHKERRA(ierr)
      call ISSetPermutation(is,ierr) 
           CHKERRA(ierr)
      call ISInvertPermutation(is,PETSC_DECIDE,newis,ierr) 
           CHKERRA(ierr)
      call ISGetIndices(newis,ii,iis,ierr) 
           CHKERRA(ierr)
      do 40, i=1,n
        if (ii(iis+i) .ne. n - i) then
          print*, 'Error getting permutation indices'
          stop
       endif
 40   continue
      call ISRestoreIndices(newis,ii,iis,ierr) 
           CHKERRA(ierr)
      call ISDestroy(newis,ierr) 
           CHKERRA(ierr)
      call ISDestroy(is,ierr) 
           CHKERRA(ierr)
      call PetscFinalize(ierr)
      end
 






