
C      "$Id: ex3.F,v 1.4 1996/07/08 22:16:12 bsmith Exp $";

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/is.h"
#include "include/FINCLUDE/viewer.h"

C
C  This example demonstrates how a Fortran user can directly access the
C  array storing a PETSc index set with ISGetIndices().  The user
C  declares an array (vss(1)) and index variable (iss), which are then used
C  together to manipulate the array contents.  The example merely prints
C  the index set array -- not really very interesting, but a decent test.
C

      integer i, n, ierr, iss, vss(1), first, step
      IS      set

#define ss(ib)  vss(iss + (ib))

      call PetscInitialize(PETSC_NULL_CHAR,ierr)
      n     = 10
      first = 3
      step  = 2

C  Create index set
      call ISCreateStrideSeq(MPI_COMM_SELF,n,first,step,set,ierr)
      call ISView(set,VIEWER_STDOUT_SELF,ierr)

C  Extract indices from set
      call ISGetIndices(set,vss,iss,ierr)
      write(6,20)
      do 10 i=1,n
         write(6,30) ss(i)
 10   continue
 20   format('print indices directly')
 30   format(i3)

C  Clean up 
      call ISRestoreIndices(set,vss,iss,ierr)
      call ISDestroy(set,ierr)
      call PetscFinalize(ierr)

      stop
      end

