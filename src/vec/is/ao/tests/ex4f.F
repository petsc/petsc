!
!    Test AO with on IS with 0 entries - Fortran version of ex4.c
!
      program main
#include <petsc/finclude/petscao.h>
      use petscao
      implicit none

      PetscErrorCode ierr
      AO             ao
      PetscInt       localvert(4),nlocal
      PetscMPIInt    rank
      IS             is
      PetscInt       one,zero

!  Needed to work with 64 bit integers from Fortran
      one  = 1
      zero = 0

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr)

      nlocal = 0
      if (rank .eq. 0) then
         nlocal = 4
         localvert(1) = 0
         localvert(2) = 1
         localvert(3) = 2
         localvert(4) = 3
      endif

!     Test AOCreateBasic()
      call AOCreateBasic(PETSC_COMM_WORLD, nlocal, localvert,             &
     &     PETSC_NULL_INTEGER,ao,ierr)
      call AODestroy(ao,ierr)

!     Test AOCreateMemoryScalable()
      call AOCreateMemoryScalable(PETSC_COMM_WORLD, nlocal, localvert,    &
     &     PETSC_NULL_INTEGER,ao,ierr)
      call AODestroy(ao,ierr)

      call AOCreate(PETSC_COMM_WORLD,ao,ierr)
      call ISCreateStride(PETSC_COMM_WORLD,one,zero,one,is,ierr)
      call AOSetIS(ao,is,is,ierr)
      call AOSetType(ao,AOMEMORYSCALABLE,ierr)
      call ISDestroy(is,ierr)
      call AODestroy(ao,ierr)

      call PetscFinalize(ierr)
      end



!/*TEST
!
!   test:
!     output_file: output/ex4_1.out
!
!   test:
!      suffix: 2
!      nsize: 2
!      output_file: output/ex4_1.out
!
!TEST*/
