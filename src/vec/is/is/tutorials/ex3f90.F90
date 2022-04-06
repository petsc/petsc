!
!    Description:  Creates an index set based on blocks of integers. Views that index set
!    and then destroys it.
!
!
      program main
#include <petsc/finclude/petscis.h>
      use petscis
      implicit none

      PetscErrorCode ierr
      PetscInt n,bs,issize
      PetscInt inputindices(4)
      PetscInt, pointer :: indices(:)
      IS       set
      PetscBool  isablock;

      n               = 4
      bs              = 3
      inputindices(1) = 0
      inputindices(2) = 1
      inputindices(3) = 3
      inputindices(4) = 4

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
         print*,'Unable to initialize PETSc'
         stop
       endif
!
!    Create a block index set. The index set has 4 blocks each of size 3.
!    The indices are {0,1,2,3,4,5,9,10,11,12,13,14}
!    Note each processor is generating its own index set
!    (in this case they are all identical)
!
      call ISCreateBlock(PETSC_COMM_SELF,bs,n,inputindices,PETSC_COPY_VALUES,set,ierr);CHKERRA(ierr)
      call ISView(set,PETSC_VIEWER_STDOUT_SELF,ierr);CHKERRA(ierr)

!
!    Extract indices from set.
!
      call ISGetLocalSize(set,issize,ierr);CHKERRA(ierr)
      call ISGetIndicesF90(set,indices,ierr);CHKERRA(ierr)
      write(6,100)indices
 100  format(12I3)
      call ISRestoreIndicesF90(set,indices,ierr);CHKERRA(ierr)

!
!    Extract the block indices. This returns one index per block.
!
      call ISBlockGetIndicesF90(set,indices,ierr);CHKERRA(ierr)
      write(6,200)indices
 200  format(4I3)
      call ISBlockRestoreIndicesF90(set,indices,ierr);CHKERRA(ierr)

!
!    Check if this is really a block index set
!
      call PetscObjectTypeCompare(set,ISBLOCK,isablock,ierr);CHKERRA(ierr)
      if (.not. isablock) then
        write(6,*) 'Index set is not blocked!'
      endif

!
!    Determine the block size of the index set
!
      call ISGetBlockSize(set,bs,ierr);CHKERRA(ierr)
      if (bs .ne. 3) then
        write(6,*) 'Blocksize != 3'
      endif

!
!    Get the number of blocks
!
      call ISBlockGetLocalSize(set,n,ierr);CHKERRA(ierr)
      if (n .ne. 4) then
        write(6,*) 'Number of blocks != 4'
      endif

      call ISDestroy(set,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!      filter: sort -b
!      filter_output: sort -b
!
!TEST*/
