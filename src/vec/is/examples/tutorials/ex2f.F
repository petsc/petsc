!
!     Description: Creates an index set based on a stride. Views that
!     index set and then destroys it.
!
!/*T
!     Concepts: index sets^manipulating a stride index set;
!     Concepts: index sets^accessing indices from Fortran
!T*/
!    
!     Include petscis.h so we can use PETSc IS objects.
!
      program main
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscis.h>

      PetscErrorCode ierr
      PetscInt    i,n,index(1),first,step,val
      IS          set
      PetscOffset iss

#define indices(ib)  index(iss + (ib))

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      n     = 10
      first = 3
      step  = 2

!     Create stride index set, starting at 3 with a stride of 2 Note
!     each processor is generating its own index set (in this case they
!     are all identical)

      call ISCreateStride(PETSC_COMM_SELF,n,first,step,set,ierr)
      call ISView(set,PETSC_VIEWER_STDOUT_SELF,ierr)

!     Extract the indice values from the set. Demonstrates how a Fortran
!     code can directly access the array storing a PETSc index set with
!     ISGetIndices().  The user declares an array (index(1)) and index
!     variable (iss), which are then used together to allow the Fortran
!     to directly manipulate the PETSc array

      call ISGetIndices(set,index,iss,ierr)
      write(6,20)
!     Bug in IRIX64 f90 compiler - write cannot handle
!     integer(integer*8) correctly
      do 10 i=1,n
         val = indices(i)
         write(6,30) val
 10   continue
 20   format('Printing indices directly')
 30   format(i3)
      call ISRestoreIndices(set,index,iss,ierr)

!     Determine information on stride

      call ISStrideGetInfo(set,first,step,ierr)
      if (first .ne. 3 .or. step .ne. 2) then
        print*,'Stride info not correct!'
      endif

      call ISDestroy(set,ierr)
      call PetscFinalize(ierr)
      end

