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

      PetscCallA(PetscInitialize(ierr))
      m = 3
      n = 2
      one = 1
!
!      Create a parallel dense matrix shared by all processors
!
      PetscCallA(MatCreateDense(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,m,n,PETSC_NULL_SCALAR,A,ierr))

!
!     Set values into the matrix. All processors set all values.
!
      do 10, i=0,m-1
        iar(1) = i
        do 20, j=0,n-1
          jar(1) = j
          v(1)   = 9.0/real(i+j+1)
          PetscCallA(MatSetValues(A,one,iar,one,jar,v,INSERT_VALUES,ierr))
 20     continue
 10   continue

      PetscCallA(MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY,ierr))
      PetscCallA(MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY,ierr))

!
!       Print the matrix to the screen
!
      PetscCallA(MatView(A,PETSC_VIEWER_STDOUT_WORLD,ierr))

!
!      Print the local matrix shape to the screen for each rank
!
      PetscCallA(MatDenseGetArrayF90(A,array,ierr))
      PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
      ashape = shape(array)
      write(string, '("[", i0, "]", " shape (", i0, ",", i0, ")", a1)') rank, ashape(1), ashape(2), new_line('a')
      PetscCallA(PetscSynchronizedPrintf(PETSC_COMM_WORLD, string, ierr))
      PetscCallA(PetscSynchronizedFlush(PETSC_COMM_WORLD,PETSC_STDOUT,ierr))
      PetscCallA(MatDenseRestoreArrayF90(A,array,ierr))
!
!      Free the space used by the matrix
!
      PetscCallA(MatDestroy(A,ierr))
      PetscCallA(PetscFinalize(ierr))
      end

!/*TEST
!
!   test:
!      nsize: 2
!      filter: sort -b
!      filter_output: sort -b
!
!TEST*/
