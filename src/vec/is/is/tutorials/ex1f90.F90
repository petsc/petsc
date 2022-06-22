!
!  Description: Creates an index set based on a set of integers. Views that index set
!  and then destroys it.
!
!

      program main
#include <petsc/finclude/petscis.h>
      use petscis
      implicit none

      PetscErrorCode ierr
      PetscInt indices(5),n
      PetscInt five
      PetscMPIInt rank
      PetscInt, pointer :: idx(:)
      IS      is

      five = 5
      PetscCallA(PetscInitialize(ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))

!  Create an index set with 5 entries. Each processor creates
!  its own index set with its own list of integers.

      indices(1) = rank + 1
      indices(2) = rank + 2
      indices(3) = rank + 3
      indices(4) = rank + 4
      indices(5) = rank + 5
      PetscCallA(ISCreateGeneral(PETSC_COMM_SELF,five,indices,PETSC_COPY_VALUES,is,ierr))

!  Print the index set to stdout

      PetscCallA(ISView(is,PETSC_VIEWER_STDOUT_SELF,ierr))

!  Get the number of indices in the set

      PetscCallA(ISGetLocalSize(is,n,ierr))

!   Get the indices in the index set

      PetscCallA(ISGetIndicesF90(is,idx,ierr))

      if (associated(idx)) then
         write (*,*) 'Association check passed'
      else
         write (*,*) 'Association check failed'
      endif

!   Now any code that needs access to the list of integers
!   has access to it here

      write(6,50) idx
 50   format(5I3)

      write(6,100) rank,idx(1),idx(5)
 100  format('[',i5,'] First index = ',i5,' fifth index = ',i5)

!   Once we no longer need access to the indices they should
!   returned to the system

      PetscCallA(ISRestoreIndicesF90(is,idx,ierr))

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
