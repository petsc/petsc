!
!  Simple PETSc Program to test setting error handlers from Fortran
!
      subroutine GenerateErr(ierr)
#include "include/finclude/petsc.h"
      PetscErrorCode  ierr

      call PetscError(1,1,'Error message',ierr)

      return
      end

      subroutine MyErrHandler(line,fun,file,dir,n,p,mess,ctx,ierr)
#include "include/finclude/petsc.h"
      integer line,n,p
      PetscInt ctx
      PetscErrorCode ierr
      character*(*) fun,file,dir,mess

      print*,'My error handler ',mess
      return
      end

      program main
#include "include/finclude/petsc.h"
      PetscErrorCode ierr
      external       MyErrHandler

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

      call PetscPushErrorHandler(PetscTraceBackErrorHandler,               &
     &                           PETSC_NULL_INTEGER,ierr)

      call GenerateErr(ierr)

      call PetscPushErrorHandler(MyErrHandler,                           &
     &                           PETSC_NULL_INTEGER,ierr)

      call GenerateErr(ierr)

      call PetscPushErrorHandler(PetscAbortErrorHandler,                   &
     &                           PETSC_NULL_INTEGER,ierr)

      call GenerateErr(ierr)

      call PetscFinalize(ierr)
      end
