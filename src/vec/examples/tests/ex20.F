!
!    "$Id: ex20.F,v 1.8 1999/01/31 16:04:09 bsmith Exp bsmith $";
!
       implicit none
#include "include/finclude/petsc.h"
#include "include/finclude/sys.h"
#include "include/finclude/vec.h"
#include "include/finclude/viewer.h"
!
!  This example demonstrates writing an array to a file in binary format
!  that may be read in by PETSc's VecLoad() routine.
!
       integer          n,ierr,i,fd,veccookie
       Scalar           array(5)
       Vec              x
       Viewer           v

       n         = 5
       veccookie = 1211211 + 3

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)                  

       do 10, i=1,5
         array(i) = i
 10    continue

!   Open binary file for writing
       call PetscBinaryOpen('testfile',BINARY_CREATE,fd,ierr)
!   Write the Vec header
       call PetscBinaryWrite(fd,veccookie,1,PETSC_INT,0,ierr)
!   Write the array length
       call PetscBinaryWrite(fd,n,1,PETSC_INT,0,ierr)
!   Write the array
       call PetscBinaryWrite(fd,array,n,PETSC_DOUBLE,0,ierr)
!   Close the file
       call PetscBinaryClose(fd,ierr)

!
!  Open the file for reading by PETSc
!
       call ViewerBinaryOpen(PETSC_COMM_SELF,'testfile',                &
     &                       BINARY_RDONLY,v,ierr)
!
!  Load the vector
!
       Call VecLoad(v,x,ierr)
       call ViewerDestroy(v,ierr)
!
!  Print the vector
!
       call VecView(x,VIEWER_STDOUT_SELF,ierr)
!

       call VecDestroy(x,ierr)
       call PetscFinalize(ierr)
       end

 
