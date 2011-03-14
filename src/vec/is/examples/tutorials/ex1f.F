!
!  Description: Creates an index set based on a set of integers. Views that index set
!  and then destroys it.
!
!/*T
!    Concepts: index sets^manipulating a general index set;
!T*/
!    
!
!  The following include statements are required for Fortran programs
!  that use PETSc index sets:
!     petscsys.h  - base PETSc routines
!     petscis.h     - index sets (IS objects)
!
      program main
      implicit none

#include <finclude/petscsys.h>
#include <finclude/petscis.h>

      PetscErrorCode ierr
      PetscInt indices(5),n,index1,index5
      PetscMPIInt rank
      PetscOffset ix
      IS          is

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

!  Create an index set with 5 entries. Each processor creates
!  its own index set with its own list of integers.
   
      indices(1) = rank + 1
      indices(2) = rank + 2 
      indices(3) = rank + 3 
      indices(4) = rank + 4 
      indices(5) = rank + 5 

!     if using 64bit integers cannot pass 5 into routine expecting an integer*8
      n = 5
      call ISCreateGeneral(PETSC_COMM_SELF,n,indices,PETSC_COPY_VALUES,       &
     &                     is,ierr)

!  Print the index set to stdout

      call ISView(is,PETSC_VIEWER_STDOUT_SELF,ierr)

!  Get the number of indices in the set 

      call ISGetLocalSize(is,n,ierr)

!   Get the indices in the index set

      call ISGetIndices(is,indices,ix,ierr)

!   Now any code that needs access to the list of integers
!   has access to it here

!
!      Bug in IRIX64-F90 libraries - write/format cannot handle integer(integer*8 + integer)
!

      index1 = indices(ix+1)
      index5 = indices(ix+5)
      write(6,100) rank,index1,index5
 100  format('[',i5,'] First index = ',i5,' fifth index = ',i5)
      
!   Once we no longer need access to the indices they should 
!   returned to the system 

      call ISRestoreIndices(is,indices,ix,ierr)
  
!   All PETSc objects should be destroyed once they are 
!   no longer needed

      call ISDestroy(is,ierr)
      call PetscFinalize(ierr)
      end

 



