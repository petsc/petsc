!
!  Description: Creates an index set based on a set of integers. Views that index set
!  and then destroys it.
!
#include <petsc/finclude/petscis.h>
program main
  use petscis
  implicit none

  PetscErrorCode ierr
  PetscInt, parameter :: n = 5
  PetscInt indices(n), i, n_out
  PetscMPIInt rank
  IS is
  PetscInt, pointer :: indices2(:)

  PetscCallA(PetscInitialize(ierr))
  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD, rank, ierr))

! Create an index set with 5 entries. Each processor creates
! its own index set with its own list of integers.
  indices = rank + [(i, i=1, n)]
  PetscCallA(ISCreateGeneral(PETSC_COMM_SELF, n, indices, PETSC_COPY_VALUES, is, ierr))

! Print the index set to stdout
  PetscCallA(ISView(is, PETSC_VIEWER_STDOUT_SELF, ierr))

! Get the number of indices in the set
  PetscCallA(ISGetLocalSize(is, n_out, ierr))

!  Get the indices in the index set
  PetscCallA(ISGetIndices(is, indices2, ierr))

! Now any code that needs access to the list of integers
! has access to it here
  write (6, 100) rank, indices(1), indices(n_out)
100 format('[', i5, '] First index = ', i5, ' fifth index = ', i5)

! Once we no longer need access to the indices they should
! returned to the system
  PetscCallA(ISRestoreIndices(is, indices2, ierr))

! All PETSc objects should be destroyed once they are
! no longer needed
  PetscCallA(ISDestroy(is, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      filter: sort -b
!      filter_output: sort -b
!
!TEST*/
