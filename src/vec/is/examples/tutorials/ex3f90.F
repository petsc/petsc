C
C      "$Id: ex3f90.F,v 1.4 1997/04/10 00:00:03 bsmith Exp balay $"; */
C
C
C/*T
C    Concepts: Index sets^Manipulating a block index set;
C    Concepts: Fortran90;
C    Routines: ISCreateBlock(); ISDestroy(); ISView()
C    Routines: ISGetIndicesF90(); ISRestoreIndicesF90(); ISBlockGetSize()
C    Routines: ISBlockGetBlockSize(); ISBlockGetIndicesF90(); ISBlockRestoreIndicesF90()
C    Routines: ISBlock()
C
C    Comment:  Creates an index set based on blocks of integers. Views that index set
C    and then destroys it.
CT*/
C
C  The following include statements are required for Fortran programs
C  that use PETSc index sets:
C     petsc.h  - base PETSc routines
C     is.h     - index sets (IS objects)
C     is90.h   - to allow access to Fortran90 features of index sets
C
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
      
C
C    Create a block index set. The index set has 4 blocks each of size 3.
C    The indices are {0,1,2,3,4,5,9,10,11,12,13,14}
C    Note each processor is generating its own index set 
C    (in this case they are all identical)
C
      call ISCreateBlock(PETSC_COMM_SELF,bs,n,inputindices,set,ierr) 
      call ISView(set,VIEWER_STDOUT_SELF,ierr) 

C
C    Extract indices from set.
C
      call ISGetSize(set,issize,ierr) 
      call ISGetIndicesF90(set,indices,ierr) 
      print*,indices
      call ISRestoreIndicesF90(set,indices,ierr) 

C
C    Extract the block indices. This returns one index per block.
C
      call ISBlockGetIndicesF90(set,indices,ierr) 
      print*, indices
      call ISBlockRestoreIndicesF90(set,indices,ierr) 

C
C    Check if this is really a block index set
C
      call ISBlock(set,isablock,ierr) 
      if (isablock .ne. PETSC_TRUE) then
        print*,'Index set is not blocked!'
      endif

C
C    Determine the block size of the index set
C
      call ISBlockGetBlockSize(set,bs,ierr) 
      if (bs .ne. 3) then
        print*,'Blocksize != 3'
      endif

C
C    Get the number of blocks
C
      call ISBlockGetSize(set,n,ierr) 
      if (n .ne. 4) then
        print*,'Number of blocks != 4'
      endif

      call ISDestroy(set,ierr) 
      call PetscFinalize(ierr)
      end


