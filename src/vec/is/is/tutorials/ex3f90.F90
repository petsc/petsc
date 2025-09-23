!
!    Description:  Creates an index set based on blocks of integers. Views that index set
!    and then destroys it.
!
program main
#include <petsc/finclude/petscis.h>
  use petscis
  implicit none

  PetscErrorCode ierr
  PetscInt n, bs, issize
  PetscInt inputindices(4)
  PetscInt, pointer :: indices(:)
  IS set
  PetscBool isablock

  n = 4
  bs = 3
  inputindices(1) = 0
  inputindices(2) = 1
  inputindices(3) = 3
  inputindices(4) = 4

  PetscCallA(PetscInitialize(ierr))
!
!    Create a block index set. The index set has 4 blocks each of size 3.
!    The indices are {0,1,2,3,4,5,9,10,11,12,13,14}
!    Note each processor is generating its own index set
!    (in this case they are all identical)
!
  PetscCallA(ISCreateBlock(PETSC_COMM_SELF, bs, n, inputindices, PETSC_COPY_VALUES, set, ierr))
  PetscCallA(ISView(set, PETSC_VIEWER_STDOUT_SELF, ierr))

!
!    Extract indices from set.
!
  PetscCallA(ISGetLocalSize(set, issize, ierr))
  PetscCallA(ISGetIndices(set, indices, ierr))
  write (6, 100) indices
100 format(12I3)
  PetscCallA(ISRestoreIndices(set, indices, ierr))

!
!    Extract the block indices. This returns one index per block.
!
  PetscCallA(ISBlockGetIndices(set, indices, ierr))
  write (6, 200) indices
200 format(4I3)
  PetscCallA(ISBlockRestoreIndices(set, indices, ierr))

!
!    Check if this is really a block index set
!
  PetscCallA(PetscObjectTypeCompare(PetscObjectCast(set), ISBLOCK, isablock, ierr))
  if (.not. isablock) then
    write (6, *) 'Index set is not blocked!'
  end if

!
!    Determine the block size of the index set
!
  PetscCallA(ISGetBlockSize(set, bs, ierr))
  if (bs /= 3) then
    write (6, *) 'Blocksize != 3'
  end if

!
!    Get the number of blocks
!
  PetscCallA(ISBlockGetLocalSize(set, n, ierr))
  if (n /= 4) then
    write (6, *) 'Number of blocks != 4'
  end if

  PetscCallA(ISDestroy(set, ierr))
  PetscCallA(PetscFinalize(ierr))
end

!/*TEST
!
!   test:
!      filter: sort -b
!      filter_output: sort -b
!
!TEST*/
