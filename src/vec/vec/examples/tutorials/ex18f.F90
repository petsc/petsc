! Computes the integral of 2*x/(1+x^2) from x=0..1
! This is equal to the ln(2).

! Concepts: vectors; assembling vectors;
! Processors: n

! Contributed by Mike McCourt <mccomic@iit.edu> and Nathan Johnston <johnnat@iit.edu>
! Fortan tranlation by Arko Bhattacharjee <a.bhattacharjee@mpie.de>


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
  character(len=128) :: output
  PetscInt,parameter :: zero = 0, one = 1, two = 2

  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
  if (ierr /= 0) then
    print*,'PetscInitialize failed'
    stop
  endif

  call MPI_Comm_rank(PETSC_COMM_WORLD,rank,ierr);CHKERRA(ierr)
  call MPI_Comm_size(PETSC_COMM_WORLD,mySize,ierr);CHKERRA(ierr)

 
  ! Create a parallel vector.
  ! Here we set up our x vector which will be given values below.
  ! The xend vector is a dummy vector to find the value of the
  ! elements at the endpoints for use in the trapezoid rule.
 
  call VecCreate(PETSC_COMM_WORLD,x,ierr);CHKERRA(ierr)
  call VecSetSizes(x,PETSC_DECIDE,numPoints,ierr);CHKERRA(ierr)
  call VecSetFromOptions(x,ierr);CHKERRA(ierr)
  call VecGetSize(x,N,ierr);CHKERRA(ierr)
  call VecSet(x,myResult,ierr);CHKERRA(ierr)
  call VecDuplicate(x,xend,ierr);CHKERRA(ierr)
  myResult = 0.5
  if (rank==0) then
    i = 0
    call VecSetValues(xend,one,i,myResult,INSERT_VALUES,ierr);CHKERRA(ierr)
  endif
  
  if (rank == mySize-1) then
    i    = N-1
    call VecSetValues(xend,one,i,myResult,INSERT_VALUES,ierr);CHKERRA(ierr)
  endif
  
  ! Assemble vector, using the 2-step process:
  ! VecAssemblyBegin(), VecAssemblyEnd()
  ! Computations can be done while messages are in transition
  ! by placing code between these two statements.

  call VecAssemblyBegin(xend,ierr);CHKERRA(ierr)
  call VecAssemblyEnd(xend,ierr);CHKERRA(ierr)


  ! Set the x vector elements.
  ! i*h will return 0 for i=0 and 1 for i=N-1.
  ! The function evaluated (2x/(1+x^2)) is defined above.
  ! Each evaluation is put into the local array of the vector without message passing.

  call VecGetOwnershipRange(x,rstart,rend,ierr);CHKERRA(ierr)
  call VecGetArrayF90(x,xarray,ierr);CHKERRA(ierr)
  k = 1
  do i=rstart,rend-1
    xarray(k) = real(i)*h
    xarray(k) = func(xarray(k))
    k = k+1
  end do
  call VecRestoreArrayF90(x,xarray,ierr);CHKERRA(ierr)
    
  
  ! Evaluates the integral.  First the sum of all the points is taken.
  ! That result is multiplied by the step size for the trapezoid rule.
  ! Then half the value at each endpoint is subtracted,
   !this is part of the composite trapezoid rule.
 
  
  call VecSum(x,myResult,ierr);CHKERRA(ierr)
  myResult = myResult*h
  call VecDot(x,xend,dummy,ierr);CHKERRA(ierr)
  myResult = myResult-h*dummy
  

 
  !Return the value of the integral.
  
 
  write(output,'(a,f9.6,a)') 'ln(2) is',real(myResult),'\n'           ! PetscScalar might be complex
  call PetscPrintf(PETSC_COMM_WORLD,trim(output),ierr);CHKERRA(ierr)
  call VecDestroy(x,ierr);CHKERRA(ierr)
  call VecDestroy(xend,ierr);CHKERRA(ierr)
 

  call PetscFinalize(ierr);CHKERRA(ierr)
  
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
