!
!
!  Tests MatDenseGetArray()
!

      program main
#include <petsc/finclude/petscmat.h>
      use petscmat
      implicit none

      Mat A
      PetscErrorCode ierr
      PetscInt i,j,m,n,iar(1),jar(1)
      PetscInt one
      PetscScalar  v(1)
      PetscScalar, pointer :: array(:,:)
      PetscMPIInt rank
      integer :: ashape(2)
      character(len=80) :: string

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif

      m = 3
      n = 2
      one = 1
!
!      Create a parallel dense matrix shared by all processors
!
      call MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_NULL_SCALAR,A,ierr);CHKERRA(ierr)

!
!     Set values into the matrix. All processors set all values.
!
      do 10, i=0,m-1
        iar(1) = i
        do 20, j=0,n-1
          jar(1) = j
          v(1)   = 9.0/real(i+j+1)
          call MatSetValues(A,one,iar,one,jar,v,INSERT_VALUES,ierr);CHKERRA(ierr)
 20     continue
 10   continue

      call MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)
      call MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr);CHKERRA(ierr)

!
!       Print the matrix to the screen
!
      call MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr);CHKERRA(ierr)

!
!      Print the local matrix shape to the screen for each rank
!
      call MatDenseGetArrayF90(A,array,ierr);CHKERRA(ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);CHKERRA(ierr)
      ashape = shape(array)
      write(string, '("[", i0, "]", " shape (", i0, ",", i0, ")", a1)') rank, ashape(1), ashape(2), new_line('a')
      call PetscSynchronizedPrintf(PETSC_COMM_WORLD, string, ierr);CHKERRA(ierr)
      call PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT,ierr);CHKERRA(ierr)
      call MatDenseRestoreArrayF90(A,array,ierr);CHKERRA(ierr)
!
!      Free the space used by the matrix
!
      call MatDestroy(A,ierr);CHKERRA(ierr)
      call PetscFinalize(ierr)
      end

!/*TEST
!
!   test:
!      nsize: 2
!      filter: sort -b
!      filter_output: sort -b
!
!TEST*/
