!
!   This program demonstrates use of MatCreateSubMatrices() from Fortran
!
      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Mat             A,B(2)
      PetscErrorCode  ierr
      PetscInt        nis,zero(1)
      PetscViewer     v
      IS              isrow
      PetscMPIInt     rank

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

#if defined(PETSC_USE_64BIT_INDICES)
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,                          &
     & '${PETSC_DIR}/share/petsc/datafiles/matrices/' //                       &
     & 'ns-real-int64-float64',                                               &
     &                          FILE_MODE_READ,v,ierr)
#else
      call PetscViewerBinaryOpen(PETSC_COMM_WORLD,                          &
     & '${PETSC_DIR}/share/petsc/datafiles/matrices/' //                       &
     & 'ns-real-int32-float64',                                               &
     &                          FILE_MODE_READ,v,ierr)
#endif

      call MatCreate(PETSC_COMM_WORLD,A,ierr)
      call MatSetType(A, MATAIJ,ierr)
      call MatLoad(A,v,ierr)

      nis     = 1
      zero(1) = 0
      if (rank .eq. 1) then
         nis = 0 ! test nis = 0
      endif
      call ISCreateGeneral(PETSC_COMM_SELF,nis,zero,PETSC_COPY_VALUES,    &
     &                     isrow,ierr)

      call MatCreateSubmatrices(A,nis,isrow,isrow,                           &
     &        MAT_INITIAL_MATRIX,B,ierr)

      if (rank .eq. 0) then
         call MatView(B(1),PETSC_VIEWER_STDOUT_SELF,ierr)
      endif

      call MatCreateSubmatrices(A,nis,isrow,isrow,                           &
     &        MAT_REUSE_MATRIX,B,ierr)

      if (rank .eq. 0) then
         call MatView(B(1),PETSC_VIEWER_STDOUT_SELF,ierr)
      endif

      call ISDestroy(isrow,ierr)
      call MatDestroy(A,ierr)
      call MatDestroySubMatrices(nis,B,ierr)
      call PetscViewerDestroy(v,ierr)

      call PetscFinalize(ierr)
      end

!/*TEST
!
!     test:
!        requires: double !complex
!
!TEST*/




