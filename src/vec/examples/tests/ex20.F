
C    "$Id: ex20.F,v 1.2 1997/07/29 14:07:43 bsmith Exp bsmith $";

#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/sys.h"
#include "include/FINCLUDE/vec.h"
#include "include/FINCLUDE/viewer.h"
C
C  This example demonstrates writing an array to a file in binary format
C  that may be read in by PETSc's VecLoad() routine.
C
       implicit none
       integer          n, ierr,i,fd,veccookie
       Scalar           array(5)
       Vec              x
       Viewer           v

       n         = 5
       veccookie = 1211211 + 3

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)                  

       do 10, i=1,5
         array(i) = i
 10    continue

C   Open binary file for writing
       call PetscBinaryOpen('testfile',BINARY_CREATE,fd,ierr)
C   Write the Vec header
       call PetscBinaryWrite(fd,veccookie,1,PETSC_INT,0,ierr)
C   Write the array length
       call PetscBinaryWrite(fd,n,1,PETSC_INT,0,ierr)
C   Write the array
       call PetscBinaryWrite(fd,array,n,PETSC_DOUBLE,0,ierr)
C   Close the file
       call PetscBinaryClose(fd,ierr)

C
C  Open the file for reading by PETSc
C
       call ViewerFileOpenBinary(PETSC_COMM_SELF,'testfile',PETSC_RDONLY,
     &                           v,ierr)
C
C  Load the vector
C
       Call VecLoad(v,x,ierr)
       call ViewerDestroy(v,ierr)
C
C  Print the vector
C
       call VecView(x,VIEWER_STDOUT_SELF,ierr)
C

       call VecDestroy(x,ierr)
       call PetscFinalize(ierr)
       end

 
