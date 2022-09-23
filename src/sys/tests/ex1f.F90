!
!  Simple PETSc Program to test setting error handlers from Fortran
!
      subroutine GenerateErr(line,ierr)

#include <petsc/finclude/petscsys.h>
      use petscsys
      PetscErrorCode  ierr
      integer line

      call PetscError(PETSC_COMM_SELF,1,PETSC_ERROR_INITIAL,'Error message')
      return
      end

      subroutine MyErrHandler(comm,line,fun,file,n,p,mess,ctx,ierr)
      use petscsysdef
      integer line,n,p
      PetscInt ctx
      PetscErrorCode ierr
      MPI_Comm comm
      character*(*) fun,file,mess

      write(6,*) 'My error handler ',mess
      call flush(6)
      return
      end

      program main
      use petscsys
      PetscErrorCode ierr
      external       MyErrHandler

      PetscCallA(PetscInitialize(ierr))
      PetscCallA(PetscPushErrorHandler(PetscTraceBackErrorHandler,PETSC_NULL_INTEGER,ierr))
      PetscCallA(GenerateErr(__LINE__,ierr))
      PetscCallA(PetscPushErrorHandler(MyErrHandler,PETSC_NULL_INTEGER,ierr))
      PetscCallA(GenerateErr(__LINE__,ierr))
      PetscCallA(PetscPushErrorHandler(PetscAbortErrorHandler,PETSC_NULL_INTEGER,ierr))
      PetscCallA(GenerateErr(__LINE__,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!
!     These test fails on some systems randomly due to the Fortran and C output becoming mixed up,
!     using a Fortran flush after the Fortran print* does not resolve the issue
!
!/*TEST
!
!   test:
!     args: -error_output_stdout
!     filter:Error: grep -E  "(My error handler|Operating system error: Cannot allocate memory)" | wc -l
!
!TEST*/
