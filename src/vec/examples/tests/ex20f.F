!
!    "$Id: ex20.F,v 1.13 2001/08/07 03:02:26 balay Exp $";
!
      program main
      implicit none
#include "include/finclude/petsc.h"
#include "include/finclude/petscsys.h"
#include "include/finclude/petscvec.h"
#include "include/finclude/petscviewer.h"
!
!  This example demonstrates writing an array to a file in binary format
!  that may be read in by PETSc's VecLoad() routine.
!
       integer          n,ierr,i,fd,veccookie(1)
       PetscScalar      array(5)
       Vec              x
       PetscViewer           v

       n            = 5
       veccookie(1) = 1211211 + 3

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)                  

       do 10, i=1,5
         array(i) = i
 10    continue

!   Open binary file for writing
       call PetscBinaryOpen('testfile',PETSC_FILE_CREATE,fd,ierr)
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
       call PetscViewerBinaryOpen(PETSC_COMM_SELF,'testfile',                &
     &                       PETSC_FILE_RDONLY,v,ierr)
!
!  Load the vector
!
       Call VecLoad(v,PETSC_NULL_CHARACTER,x,ierr)
       call PetscViewerDestroy(v,ierr)
!
!  Print the vector
!
       call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr)
!

       call VecDestroy(x,ierr)
       call PetscFinalize(ierr)
       end

 
