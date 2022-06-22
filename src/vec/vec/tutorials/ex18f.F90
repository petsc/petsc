! Computes the integral of 2*x/(1+x^2) from x=0..1
! This is equal to the ln(2).

! Contributed by Mike McCourt <mccomic@iit.edu> and Nathan Johnston <johnnat@iit.edu>
! Fortran translation by Arko Bhattacharjee <a.bhattacharjee@mpie.de>

program main
#include <petsc/finclude/petscvec.h>
  use petscvec
  implicit none

  PetscErrorCode :: ierr
  PetscMPIInt :: rank,mySize
  PetscInt   ::  rstart,rend,i,k,N
  PetscInt, parameter   ::   numPoints=1000000
  PetscScalar  ::  dummy
  PetscScalar, parameter  :: h=1.0/numPoints
  PetscScalar, pointer, dimension(:)  :: xarray
  PetscScalar :: myResult = 0
  Vec            x,xend
  character(len=PETSC_MAX_PATH_LEN) :: output
  PetscInt,parameter :: zero = 0, one = 1, two = 2

  PetscCallA(PetscInitialize(ierr))

  PetscCallMPIA(MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr))
  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,mySize,ierr))

  ! Create a parallel vector.
  ! Here we set up our x vector which will be given values below.
  ! The xend vector is a dummy vector to find the value of the
  ! elements at the endpoints for use in the trapezoid rule.

  PetscCallA(VecCreate(PETSC_COMM_WORLD,x,ierr))
  PetscCallA(VecSetSizes(x,PETSC_DECIDE,numPoints,ierr))
  PetscCallA(VecSetFromOptions(x,ierr))
  PetscCallA(VecGetSize(x,N,ierr))
  PetscCallA(VecSet(x,myResult,ierr))
  PetscCallA(VecDuplicate(x,xend,ierr))
  myResult = 0.5
  if (rank==0) then
    i = 0
    PetscCallA(VecSetValues(xend,one,i,myResult,INSERT_VALUES,ierr))
  endif

  if (rank == mySize-1) then
    i    = N-1
    PetscCallA(VecSetValues(xend,one,i,myResult,INSERT_VALUES,ierr))
  endif

  ! Assemble vector, using the 2-step process:
  ! VecAssemblyBegin(), VecAssemblyEnd()
  ! Computations can be done while messages are in transition
  ! by placing code between these two statements.

  PetscCallA(VecAssemblyBegin(xend,ierr))
  PetscCallA(VecAssemblyEnd(xend,ierr))

  ! Set the x vector elements.
  ! i*h will return 0 for i=0 and 1 for i=N-1.
  ! The function evaluated (2x/(1+x^2)) is defined above.
  ! Each evaluation is put into the local array of the vector without message passing.

  PetscCallA(VecGetOwnershipRange(x,rstart,rend,ierr))
  PetscCallA(VecGetArrayF90(x,xarray,ierr))
  k = 1
  do i=rstart,rend-1
    xarray(k) = real(i)*h
    xarray(k) = func(xarray(k))
    k = k+1
  end do
  PetscCallA(VecRestoreArrayF90(x,xarray,ierr))

  ! Evaluates the integral.  First the sum of all the points is taken.
  ! That result is multiplied by the step size for the trapezoid rule.
  ! Then half the value at each endpoint is subtracted,
  ! this is part of the composite trapezoid rule.

  PetscCallA(VecSum(x,myResult,ierr))
  myResult = myResult*h
  PetscCallA(VecDot(x,xend,dummy,ierr))
  myResult = myResult-h*dummy

  !Return the value of the integral.

  write(output,'(a,f9.6,a)') 'ln(2) is',real(myResult),'\n'           ! PetscScalar might be complex
  PetscCallA(PetscPrintf(PETSC_COMM_WORLD,trim(output),ierr))
  PetscCallA(VecDestroy(x,ierr))
  PetscCallA(VecDestroy(xend,ierr))

  PetscCallA(PetscFinalize(ierr))

  contains

    function func(a)
#include <petsc/finclude/petscvec.h>
      use petscvec

      implicit none
      PetscScalar :: func
      PetscScalar,INTENT(IN) :: a

      func = 2.0*a/(1.0+a*a)

    end function func

end program

!/*TEST
!
!     test:
!       nsize: 1
!       output_file: output/ex18_1.out
!
!     test:
!       nsize: 2
!       suffix: 2
!       output_file: output/ex18_1.out
!
!TEST*/
