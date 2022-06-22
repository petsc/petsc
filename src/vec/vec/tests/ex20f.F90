!
      program main
#include <petsc/finclude/petscvec.h>
      use petscvec
      implicit none

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

       PetscCallA(PetscInitialize(ierr))

       do 10, i=1,5
         array(i) = i
 10    continue

!      Open binary file for writing
       PetscCallA(PetscBinaryOpen('testfile',FILE_MODE_WRITE,fd,ierr))
!      Write the Vec header
       PetscCallA(PetscBinaryWrite(fd,vecclassid,ione,PETSC_INT,ierr))
!      Write the array length
       PetscCallA(PetscBinaryWrite(fd,n,ione,PETSC_INT,ierr))
!      Write the array
       PetscCallA(PetscBinaryWrite(fd,array,n,PETSC_SCALAR,ierr))
!      Close the file
       PetscCallA(PetscBinaryClose(fd,ierr))

!
!      Open the file for reading by PETSc
!
       PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF,'testfile',FILE_MODE_READ,v,ierr))
!
!      Load the vector
!
       PetscCallA(VecCreate(PETSC_COMM_WORLD,x,ierr))
       PetscCallA(VecLoad(x,v,ierr))
       PetscCallA(PetscViewerDestroy(v,ierr))
!
!      Print the vector
!
       PetscCallA(VecView(x,PETSC_VIEWER_STDOUT_SELF,ierr))
!

       PetscCallA(VecDestroy(x,ierr))
       PetscCallA(PetscFinalize(ierr))
       end

!/*TEST
!
!     test:
!
!TEST*/
