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

      print*,'My error handler ',mess
      return
      end

      program main
      use petscsys
      PetscErrorCode ierr
      external       MyErrHandler

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      call PetscPushErrorHandler(PetscTraceBackErrorHandler,PETSC_NULL_INTEGER,ierr)

      call GenerateErr(__LINE__,ierr)

      call PetscPushErrorHandler(MyErrHandler,PETSC_NULL_INTEGER,ierr)

      call GenerateErr(__LINE__,ierr)

      call PetscPushErrorHandler(PetscAbortErrorHandler,PETSC_NULL_INTEGER,ierr)

      call GenerateErr(__LINE__,ierr)

      call PetscFinalize(ierr)
      end

!
!     These test fails on some systems randomly due to the Fortran and C output becoming mixxed up,
!     using a Fortran flush after the Fortran print* does not resolve the issue
!
!/*TEST
!
!   test:
!     args: -error_output_stdout
!     filter:Error: egrep  "(My error handler|Operating system error: Cannot allocate memory)" | wc -l
!
!TEST*/
