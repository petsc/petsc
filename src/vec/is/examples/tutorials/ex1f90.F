
C  "$Id: ex1f90.F,v 1.2 1997/01/22 18:41:18 bsmith Exp curfman $";
C
C/*T
C    Concepts: Index sets^Manipulating a general index set;
C    Concepts: Fortran90;
C    Routines: ISCreateGeneral(); ISDestroy(); ISView(); ISGetSize()
C    Routines: ISGetIndicesF90(); ISRestoreIndicesF90()
CT*/
C    
C  Comment: Creates an index set based on a set of integers. Views that index set
C  and then destroys it.
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

      integer ierr, indices(5),rank,n
      integer, pointer :: idx(:)
      IS      is

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

C  Create an index set with 5 entries. Each processor creates
C  its own index set with its own list of integers.
   
      indices(1) = rank + 1
      indices(2) = rank + 2 
      indices(3) = rank + 3 
      indices(4) = rank + 4 
      indices(5) = rank + 5 
      call ISCreateGeneral(MPI_COMM_SELF,5,indices,is,ierr)

C  Print the index set to stdout

      call ISView(is,VIEWER_STDOUT_SELF,ierr)

C  Get the number of indices in the set 

      call ISGetSize(is,n,ierr)

C   Get the indices in the index set

      call ISGetIndicesF90(is,idx,ierr)

C   Now any code that needs access to the list of integers
C   has access to it here

      print*, idx

      write(6,100) rank,idx(1),idx(5)
 100  format('[',i5,'] First index = ',i5,' fifth index = ',i5)
      
C   Once we no longer need access to the indices they should 
C   returned to the system 

      call ISRestoreIndicesF90(is,idx,ierr)
  
C   All PETSc objects should be destroyed once they are 
C   no longer needed

      call ISDestroy(is,ierr)
      call PetscFinalize(ierr)
      stop
      end

 



