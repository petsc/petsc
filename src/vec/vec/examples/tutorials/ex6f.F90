program main
#include <petsc/finclude/petscvec.h>
  use petscvec

  implicit none

  PetscErrorCode ierr
  PetscMPIInt ::   mySize
  integer     ::      fd
  PetscInt    ::   i,sz
  PetscInt,parameter   ::   m = 10
  PetscInt,parameter   ::   one = 1
  PetscInt, allocatable,dimension(:) :: t
  PetscScalar, pointer, dimension(:) :: avec
  PetscScalar, pointer, dimension(:) :: array
  Vec            vec
  PetscViewer    view_out,view_in
  character(len=256) :: outstring
  PetscBool :: flg
 
  call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
   if (ierr /= 0) then
   print*,'PetscInitialize failed'
   stop
  endif
  
  call MPI_Comm_size(PETSC_COMM_WORLD,mySize,ierr)
  
  if (mySize /= 1) then
  SETERRA(PETSC_COMM_SELF,1,"This is a uniprocessor example only!")
  endif
   
  

  call PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-m",m,flg,ierr);CHKERRA(ierr)

  ! ----------------------------------------------------------------------
  !          PART 1: Write some data to a file in binary format           
  ! ----------------------------------------------------------------------

  ! Allocate array and set values
  
  allocate(array(0:m-1))
  do i=0,m-1
    array(i) =  real(i)*10.0
  end do
 
  allocate(t(1))
  t(1) = m
  ! Open viewer for binary output
  call PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_WRITE,view_out,ierr);CHKERRA(ierr)
  call PetscViewerBinaryGetDescriptor(view_out,fd,ierr);CHKERRA(ierr)
  
  ! Write binary output
  call PetscBinaryWrite(fd,t,one,PETSC_INT,PETSC_FALSE,ierr);CHKERRA(ierr)
  call PetscBinaryWrite(fd,array,m,PETSC_SCALAR,PETSC_FALSE,ierr);CHKERRA(ierr)
  
  ! Destroy the output viewer and work array
  call PetscViewerDestroy(view_out,ierr);CHKERRA(ierr)
  deallocate(array)

  ! ----------------------------------------------------------------------
  !          PART 2: Read data from file and form a vector                
  ! ---------------------------------------------------------------------- 

  ! Open input binary viewer 
  call PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_READ,view_in,ierr);CHKERRA(ierr)
  call PetscViewerBinaryGetDescriptor(view_in,fd,ierr);CHKERRA(ierr)

  ! Create vector and get pointer to data space 
  call VecCreate(PETSC_COMM_SELF,vec,ierr);CHKERRA(ierr)
  call VecSetSizes(vec,PETSC_DECIDE,m,ierr);CHKERRA(ierr)
  
  call VecSetFromOptions(vec,ierr);CHKERRA(ierr)
  
  call VecGetArrayF90(vec,avec,ierr);CHKERRA(ierr)

  ! Read data into vector
  call PetscBinaryRead(fd,t,one,PETSC_NULL_INTEGER,PETSC_INT,ierr);CHKERRA(ierr)
  sz=t(1)
  
  if (sz <= 0) then
   SETERRA(PETSC_COMM_SELF,1,"Error: Must have array length > 0")
  endif
  
  write(outstring,'(a,i2.2,a)') "reading data in binary from input.dat, sz =", sz, " ...\n"
  call PetscPrintf(PETSC_COMM_SELF,trim(outstring),ierr);CHKERRA(ierr)
  
  call PetscBinaryRead(fd,avec,sz,PETSC_NULL_INTEGER,PETSC_SCALAR,ierr);CHKERRA(ierr)

  ! View vector 
  call VecRestoreArrayF90(vec,avec,ierr);CHKERRA(ierr)
  call VecView(vec,PETSC_VIEWER_STDOUT_SELF,ierr);CHKERRA(ierr)

  ! Free data structures 
  deallocate(t)
  call VecDestroy(vec,ierr);CHKERRA(ierr)
  call PetscViewerDestroy(view_in,ierr);CHKERRA(ierr)
  call PetscFinalize(ierr);CHKERRA(ierr)
  
  end program

!/*TEST
!
!     test:
!        output_file: output/ex6_1.out
!
!TEST*/
