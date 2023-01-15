!
!     Description: Creates an index set based on a stride. Views that
!     index set and then destroys it.
!
!
!     Include petscis.h so we can use PETSc IS objects.
!
      program main
#include <petsc/finclude/petscis.h>
      use petscis
      implicit none

      PetscErrorCode ierr
      PetscInt    i,n,first,step,val
      IS          set
      PetscInt, pointer :: index(:)

      PetscCallA(PetscInitialize(ierr))
      n     = 10
      first = 3
      step  = 2

!     Create stride index set, starting at 3 with a stride of 2 Note
!     each processor is generating its own index set (in this case they
!     are all identical)

      PetscCallA(ISCreateStride(PETSC_COMM_SELF,n,first,step,set,ierr))
      PetscCallA(ISView(set,PETSC_VIEWER_STDOUT_SELF,ierr))

!     Extract the indice values from the set. Demonstrates how a Fortran
!     code can directly access the array storing a PETSc index set with
!     ISGetIndicesF90().

      PetscCallA(ISGetIndicesF90(set,index,ierr))
      write(6,20)
!     Bug in IRIX64 f90 compiler - write cannot handle
!     integer(integer*8) correctly
      do 10 i=1,n
         val = index(i)
         write(6,30) val
 10   continue
 20   format('Printing indices directly')
 30   format(i3)
      PetscCallA(ISRestoreIndicesF90(set,index,ierr))

!     Determine information on stride

      PetscCallA(ISStrideGetInfo(set,first,step,ierr))
      if (first .ne. 3 .or. step .ne. 2) then
        print*,'Stride info not correct!'
      endif

      PetscCallA(ISDestroy(set,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!
!TEST*/
