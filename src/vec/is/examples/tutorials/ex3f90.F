!
!      "$Id: ex3f90.F,v 1.5 1997/07/10 15:44:26 balay Exp balay $"; */
!
!
!/*T
!    Concepts: Index sets^Manipulating a block index set;
!    Concepts: Fortran90;
!    Routines: ISCreateBlock(); ISDestroy(); ISView()
!    Routines: ISGetIndicesF90(); ISRestoreIndicesF90(); ISBlockGetSize()
!    Routines: ISBlockGetBlockSize(); ISBlockGetIndicesF90(); ISBlockRestoreIndicesF90()
!    Routines: ISBlock()
!
!    Comment:  Creates an index set based on blocks of integers. Views that index set
!    and then destroys it.
!T*/
!
!  The following include statements are required for Fortran programs
!  that use PETSc index sets:
!     petsc.h  - base PETSc routines
!     is.h     - index sets (IS objects)
!     is90.h   - to allow access to Fortran90 features of index sets
!
      implicit none

#include "FINCLUDE/petsc.h"
#include "FINCLUDE/is.h"
#include "FINCLUDE/is.h90"

      integer n, ierr,bs,issize,inputindices(4)
      integer, pointer :: indices(:)
      IS       set
      PetscTruth isablock;

      n               = 4
      bs              = 3
      inputindices(1) = 0
      inputindices(2) = 3
      inputindices(3) = 9
      inputindices(4) = 12
    
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      
!
!    Create a block index set. The index set has 4 blocks each of size 3.
!    The indices are {0,1,2,3,4,5,9,10,11,12,13,14}
!    Note each processor is generating its own index set 
!    (in this case they are all identical)
!
      call ISCreateBlock(PETSC_COMM_SELF,bs,n,inputindices,set,ierr) 
      call ISView(set,VIEWER_STDOUT_SELF,ierr) 

!
!    Extract indices from set.
!
      call ISGetSize(set,issize,ierr) 
      call ISGetIndicesF90(set,indices,ierr) 
      print*,indices
      call ISRestoreIndicesF90(set,indices,ierr) 

!
!    Extract the block indices. This returns one index per block.
!
      call ISBlockGetIndicesF90(set,indices,ierr) 
      print*, indices
      call ISBlockRestoreIndicesF90(set,indices,ierr) 

!
!    Check if this is really a block index set
!
      call ISBlock(set,isablock,ierr) 
      if (isablock .ne. PETSC_TRUE) then
        print*,'Index set is not blocked!'
      endif

!
!    Determine the block size of the index set
!
      call ISBlockGetBlockSize(set,bs,ierr) 
      if (bs .ne. 3) then
        print*,'Blocksize != 3'
      endif

!
!    Get the number of blocks
!
      call ISBlockGetSize(set,n,ierr) 
      if (n .ne. 4) then
        print*,'Number of blocks != 4'
      endif

      call ISDestroy(set,ierr) 
      call PetscFinalize(ierr)
      end


