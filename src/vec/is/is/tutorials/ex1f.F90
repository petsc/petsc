!
!  Description: Creates an index set based on a set of integers. Views that index set
!  and then destroys it.
!
!
!
      program main
#include <petsc/finclude/petscis.h>
      use petscis
      implicit none

      PetscErrorCode ierr
      PetscInt n,indices(5),index1,index5
      PetscMPIInt rank
      IS          is
      PetscInt, pointer :: indices2(:)

      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

!  Create an index set with 5 entries. Each processor creates
!  its own index set with its own list of integers.

      indices(1) = rank + 1
      indices(2) = rank + 2
      indices(3) = rank + 3
      indices(4) = rank + 4
      indices(5) = rank + 5

!     if using 64bit integers cannot pass 5 into routine expecting an integer*8
      n = 5
      PetscCallA(ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,is,ierr))

!  Print the index set to stdout

      PetscCallA(ISView(is,PETSC_VIEWER_STDOUT_SELF,ierr))

!  Get the number of indices in the set

      PetscCallA(ISGetLocalSize(is,n,ierr))

!   Get the indices in the index set

      PetscCallA(ISGetIndicesF90(is,indices2,ierr))

!   Now any code that needs access to the list of integers
!   has access to it here

!
!      Bug in IRIX64-F90 libraries - write/format cannot handle integer(integer*8 + integer)
!

      index1 = indices(1)
      index5 = indices(5)
      write(6,100) rank,index1,index5
 100  format('[',i5,'] First index = ',i5,' fifth index = ',i5)

!   Once we no longer need access to the indices they should
!   returned to the system

      PetscCallA(ISRestoreIndicesF90(is,indices2,ierr))

!   All PETSc objects should be destroyed once they are
!   no longer needed

      PetscCallA(ISDestroy(is,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      filter: sort -b
!      filter_output: sort -b
!
!TEST*/
