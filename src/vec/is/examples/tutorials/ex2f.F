
C      "$Id: ex2f.F,v 1.1 1996/08/14 15:15:49 bsmith Exp bsmith $";
C
C    Concepts: Index set, indices, stride, accessing PETSc array from Fortran
C    Routines: ISCreateStride(), ISDestroy(), ISView()
C    Routines: ISGetIndices(), ISRestoreIndices()
C    
C     Creates an index set based on a stride. Views that index set
C  and then destroys it.
C
C
C  Always include petsc.h
C  include is.h so we can work with PETSc IS objects 
C  include viewer.h so we can use Viewer (e.g. VIEWER_STDOUT_SELF)

#include "FINCLUDE/petsc.h"
#include "FINCLUDE/is.h"
#include "FINCLUDE/viewer.h"


      integer i, n, ierr, iss, index(1), first, step
      IS      set

#define indices(ib)  index(iss + (ib))

      call PetscInitialize(PETSC_NULL_CHAR,ierr)
      n     = 10
      first = 3
      step  = 2

C  Create stride index set, starting at 3 with a stride of 2
C    Note each processor is generating its own index set 
C    (in this case they are all identical)

      call ISCreateStride(MPI_COMM_SELF,n,first,step,set,ierr)
      call ISView(set,VIEWER_STDOUT_SELF,ierr)

C  Extract the indice values from the set. Demonstrates how a Fortran code can  
C  directly access the array storing a PETSc index set with ISGetIndices().  
C  The user declares an array (index(1)) and index variable (iss), which are 
C  then used together to allow the Fortran to directly manipulate the PETSc array

      call ISGetIndices(set,index,iss,ierr)
      write(6,20)
      do 10 i=1,n
         write(6,30) indices(i)
 10   continue
 20   format('Printing indices directly')
 30   format(i3)
      call ISRestoreIndices(set,index,iss,ierr)

C  Determine information on stride

      call ISStrideGetInfo(set,first,step,ierr)
      if (first .ne. 3 .or. step .ne. 2) then
        print*,'Stride info not correct!'
      endif

      call ISDestroy(set,ierr)
      call PetscFinalize(ierr)

      stop
      end

