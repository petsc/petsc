!
!
!   "Scatters from a parallel vector to a sequential vector.  In
!  this case each local vector is as long as the entire parallel vector.
!
      implicit none
#include "include/finclude/petsc.h"
#include "include/finclude/petscis.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscsys.h"

      integer      n,ierr
      integer      size,rank,NN,low,high,iglobal,i
      PetscScalar  value,zero
      Vec          x,y
      IS           is1,is2
      VecScatter   ctx

      n    = 5
      zero = 0.d0
      call  PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call MPI_Comm_size(MPI_COMM_WORLD,size,ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

!     create two vectors 
!     one parallel and one sequential. The sequential one on each processor
!     is as long as the entire parallel one.

      NN = size*n

      call VecCreateMPI(MPI_COMM_WORLD,PETSC_DECIDE,NN,y,ierr)
      call VecCreateSeq(MPI_COMM_SELF,NN,x,ierr)

      call VecSet(zero,x,ierr)
      call VecGetOwnershipRange(y,low,high,ierr)
      do 10, i=0,n-1
         iglobal = i + low
         value   = i + 10*rank
         call VecSetValues(y,1,iglobal,value,INSERT_VALUES,ierr)
 10   continue

      call VecAssemblyBegin(y,ierr)
      call VecAssemblyEnd(y,ierr)
!
!   View the parallel vector
!
      call VecView(y,PETSC_VIEWER_STDOUT_WORLD,ierr)

!     create two index sets and the scatter context to move the contents of
!     of the parallel vector to each sequential vector. If you want the
!     parallel vector delivered to only one processor then create a is2 
!     of length zero on all processors except the one to receive the parallel vector

      call ISCreateStride(PETSC_COMM_SELF,NN,0,1,is1,ierr)
      call ISCreateStride(PETSC_COMM_SELF,NN,0,1,is2,ierr)
      call VecScatterCreate(y,is2,x,is1,ctx,ierr)
      call VecScatterBegin(y,x,ADD_VALUES,SCATTER_FORWARD,ctx,ierr)
      call VecScatterEnd(y,x,ADD_VALUES,SCATTER_FORWARD,ctx,ierr)
      call VecScatterDestroy(ctx,ierr)
!
!   View the sequential vector on the 0th processor
!  
      if (rank .eq. 0) then
        call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr)
      endif

      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)
      call ISDestroy(is1,ierr)
      call ISDestroy(is2,ierr)

      call PetscFinalize(ierr) 
      end
 
