!
      program main
      implicit none
#include <finclude/petscsys.h>
#include <finclude/petscvec.h>
#include <finclude/petscviewer.h>

!
!      This example demonstrates writing an array to a file in binary
!      format that may be read in by PETSc's VecLoad() routine.
!
       PetscInt n,i,ione
       PetscErrorCode ierr
       integer fd
       PetscInt vecclassid(1)
       PetscScalar      array(5)
       Vec              x
       PetscViewer           v

       ione         = 1
       n            = 5
       vecclassid(1) = 1211211 + 3

       call PetscInitialize(PETSC_NULL_CHARACTER,ierr)                  

       do 10, i=1,5
         array(i) = i
 10    continue

!      Open binary file for writing
       call PetscBinaryOpen('testfile',FILE_MODE_WRITE,fd,ierr)
!      Write the Vec header
       call PetscBinaryWrite(fd,vecclassid,ione,PETSC_INT,                    &
     &                       PETSC_FALSE,ierr)
!      Write the array length
       call PetscBinaryWrite(fd,n,ione,PETSC_INT,PETSC_FALSE,ierr)
!      Write the array
       call PetscBinaryWrite(fd,array,n,PETSC_DOUBLE,PETSC_FALSE,ierr)
!      Close the file
       call PetscBinaryClose(fd,ierr)

!
!      Open the file for reading by PETSc
!
       call PetscViewerBinaryOpen(PETSC_COMM_SELF,'testfile',                &
     &                       FILE_MODE_READ,v,ierr)
!
!      Load the vector
!
       call VecCreate(PETSC_COMM_WORLD,x,ierr)
       call VecLoad(x,v,ierr)
       call PetscViewerDestroy(v,ierr)
!
!      Print the vector
!
       call VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr)
!

       call VecDestroy(x,ierr)
       call PetscFinalize(ierr)
       end

 
