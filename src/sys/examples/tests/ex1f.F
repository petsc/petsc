!
!  Simple PETSc Program to test setting error handlers from Fortran
!
      subroutine GenerateErr(line,ierr)
#include <finclude/petscsys.h>
      PetscErrorCode  ierr
      PetscInt line

      call PetscError(PETSC_COMM_SELF,1,line,1,'Error message',ierr)

      return
      end

      subroutine MyErrHandler(line,fun,file,dir,n,p,mess,ctx,ierr)
#include <finclude/petscsys.h>
      integer line,n,p
      PetscInt ctx
      PetscErrorCode ierr
      character*(*) fun,file,dir,mess

      print*,'My error handler ',mess
      return
      end

      program main
#include <finclude/petscsys.h>
      PetscErrorCode ierr
      external       MyErrHandler

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call PetscPushErrorHandler(PetscTraceBackErrorHandler,               &
     &                           PETSC_NULL_INTEGER,ierr)

      call GenerateErr(__LINE__,ierr)

      call PetscPushErrorHandler(MyErrHandler,                           &
     &                           PETSC_NULL_INTEGER,ierr)

      call GenerateErr(__LINE__,ierr)

      call PetscPushErrorHandler(PetscAbortErrorHandler,                   &
     &                           PETSC_NULL_INTEGER,ierr)

      call GenerateErr(__LINE__,ierr)

      call PetscFinalize(ierr)
      end
