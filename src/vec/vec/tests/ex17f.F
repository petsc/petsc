!
!
!   "Scatters from a parallel vector to a sequential vector.  In
!  this case each local vector is as long as the entire parallel vector.
!
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

      PetscErrorCode ierr
      PetscMPIInt  size,rank
      PetscInt     n,NN,low,high
      PetscInt     iglobal,i,ione
      PetscInt     first,stride
      PetscScalar  value,zero
      Vec          x,y
      IS           is1,is2
      VecScatter   ctx

      n    = 5
      zero = 0.0
      call  PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      call MPI_Comm_size(PETSC_COMM_WORLD,size,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

!     create two vectors
!     one parallel and one sequential. The sequential one on each processor
!     is as long as the entire parallel one.

      NN = size*n

      call VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE,NN,y,ierr)
      call VecCreateSeq(PETSC_COMM_SELF,NN,x,ierr)

      call VecSet(x,zero,ierr)
      call VecGetOwnershipRange(y,low,high,ierr)
      ione = 1
      do 10, i=0,n-1
         iglobal = i + low
         value   = i + 10*rank
         call VecSetValues(y,ione,iglobal,value,INSERT_VALUES,ierr)
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

      first = 0
      stride = 1
      call ISCreateStride(PETSC_COMM_SELF,NN,first,stride,is1,ierr)
      call ISCreateStride(PETSC_COMM_SELF,NN,first,stride,is2,ierr)
      call VecScatterCreate(y,is2,x,is1,ctx,ierr)
      call VecScatterBegin(ctx,y,x,ADD_VALUES,SCATTER_FORWARD,ierr)
      call VecScatterEnd(ctx,y,x,ADD_VALUES,SCATTER_FORWARD,ierr)
      call VecScatterDestroy(ctx,ierr)
!
!   View the sequential vector on the 0th processor
!
      if (rank .eq. 0) then
        call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr)
      endif

#if defined(PETSC_HAVE_FORTRAN_TYPE_STAR)
      call PetscBarrier(y,ierr)
      call PetscBarrier(is1,ierr)
#endif
      call VecDestroy(x,ierr)
      call VecDestroy(y,ierr)
      call ISDestroy(is1,ierr)
      call ISDestroy(is2,ierr)

      call PetscFinalize(ierr)
      end


!/*TEST
!
!     test:
!       nsize: 3
!       filter:  grep -v "MPI processes"
!
!TEST*/
