!
!      "$Id: ex2f.F,v 1.10 1998/03/12 23:15:05 bsmith Exp balay $";
!
!/*T
!    Concepts: Index sets^Manipulating a stride index set;
!    Concepts: Index sets^Accessing indices from Fortran
!    Routines: ISCreateStride(); ISDestroy(); ISView()
!    Routines: ISGetIndices(); ISRestoreIndices()
!T*/
!    
!    Comment: Creates an index set based on a stride. Views that index set
!    and then destroys it.
!
!  Include is.h so we can use PETSc IS objects.
!
      implicit none
#include "FINCLUDE/petsc.h"
#include "FINCLUDE/is.h"

      integer     i, n, ierr,  index(1), first, step
      IS          set
      PetscOffset iss

#define indices(ib)  index(iss + (ib))

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      n     = 10
      first = 3
      step  = 2

!  Create stride index set, starting at 3 with a stride of 2
!    Note each processor is generating its own index set 
!    (in this case they are all identical)

      call ISCreateStride(PETSC_COMM_SELF,n,first,step,set,ierr)
      call ISView(set,VIEWER_STDOUT_SELF,ierr)

!  Extract the indice values from the set. Demonstrates how a Fortran code can  
!  directly access the array storing a PETSc index set with ISGetIndices().  
!  The user declares an array (index(1)) and index variable (iss), which are 
!  then used together to allow the Fortran to directly manipulate the PETSc array

      call ISGetIndices(set,index,iss,ierr)
      write(6,20)
      do 10 i=1,n
         write(6,30) indices(i)
 10   continue
 20   format('Printing indices directly')
 30   format(i3)
      call ISRestoreIndices(set,index,iss,ierr)

!  Determine information on stride

      call ISStrideGetInfo(set,first,step,ierr)
      if (first .ne. 3 .or. step .ne. 2) then
        print*,'Stride info not correct!'
      endif

      call ISDestroy(set,ierr)
      call PetscFinalize(ierr)
      end

