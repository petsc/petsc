C
C  "$Id: ex2f.F,v 1.4 1996/09/12 16:24:47 bsmith Exp bsmith $"
C
C  Formated Test for IS stride routines
C
      implicit none
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/is.h"

      integer    i, n, ierr,ii(1),iis,start,stride
      IS         is
      PetscTruth flag

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

C     Test IS of size 0 

      call ISCreateStride(MPI_COMM_SELF,0,0,2,is,ierr)
      call ISGetSize(is,n,ierr)
      if (n .ne. 0) then 
        SETERRA(1,0)
      endif
      call ISStrideGetInfo(is,start,stride,ierr)
      if (start .ne. 0) then 
         SETERRA(1,0)
      endif
      if (stride .ne. 2) then
        SETERRA(1,0)
      endif
      call ISStride(is,flag,ierr)
      if (flag .ne. PETSC_TRUE) then 
        SETERRA(1,0) 
      endif
      call ISGetIndices(is,ii,iis,ierr)
      call ISRestoreIndices(is,ii,iis,ierr)
      call ISDestroy(is,ierr)

C     Test ISGetIndices()

      call ISCreateStride(MPI_COMM_SELF,10000,-8,3,is,ierr)
      call ISGetSize(is,n,ierr)
      call ISGetIndices(is,ii,iis,ierr)
      do 10, i=1,10000
        if (ii(i+iis) .ne. -11 + 3*i) then 
          SETERRA(1,0)
        endif
 10   continue
      call ISRestoreIndices(is,ii,iis,ierr)
      call ISDestroy(is,ierr)

      call PetscFinalize(ierr)
      stop 
      end
 






