
C  "$Id: ex2.F,v 1.1 1996/08/14 03:10:30 bsmith Exp bsmith $";
C
C
C    Concepts: Index set, indices
C    Routines: ISCreateGeneral(), ISDestroy(), ISView(), ISGetSize()
C    Routines: ISGetIndices(), ISRestoreIndices()
C    
C     Creates an index set based on a set of integers. Views that index set
C  and then destroys it.
C
C
C  Always include petsc.h
C  include is.h so we can work with PETSc IS objects 
C  include viewer.h so we can use Viewer (e.g. VIEWER_STDOUT_SELF)

#include "FINCLUDE/petsc.h"
#include "FINCLUDE/is.h"
#include "FINCLUDE/viewer.h"

      integer ierr, indices(5),rank,n,ix
      IS      is

      call PetscInitialize(PETSC_NULL_CHAR,ierr)
      call MPI_Comm_rank(MPI_COMM_WORLD,rank,ierr)

C     Create an index set with 5 entries. Each processor creates
C   its own index set with its own list of integers.
   
      indices(1) = rank + 1
      indices(2) = rank + 2 
      indices(3) = rank + 3 
      indices(4) = rank + 4 
      indices(5) = rank + 5 
      call ISCreateGeneral(MPI_COMM_SELF,5,indices,is,ierr)

C   Print the index set to stdout

      call ISView(is,VIEWER_STDOUT_SELF,ierr)

C   Get the number of indices in the set 

      call ISGetSize(is,n,ierr)

C   Get the indices in the index set

      call ISGetIndices(is,indices,ix,ierr)

C    Now any code that needs access to the list of integers
C   has access to it here

      print*, '[',rank,']First index ',indices(ix+1),' fifth index',
     $        indices(ix+5)
      
C    Once we no longer need access to the indices they should 
C   returned to the system 

      call ISRestoreIndices(is,indices,ix,ierr)
  
C     One should destroy any PETSc object once one is completely
C    done with it.

      call ISDestroy(is,ierr)
      call PetscFinalize(ierr)
      stop
      end

 
