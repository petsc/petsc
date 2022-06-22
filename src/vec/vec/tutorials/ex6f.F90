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

  PetscCallA(PetscInitialize(ierr))

  PetscCallMPIA(MPI_Comm_size(PETSC_COMM_WORLD,mySize,ierr))

  if (mySize /= 1) then
    SETERRA(PETSC_COMM_SELF,PETSC_ERR_WRONG_MPI_SIZE,"This is a uniprocessor example only!")
  endif

  PetscCallA(PetscOptionsGetInt(PETSC_NULL_OPTIONS,PETSC_NULL_CHARACTER,"-m",m,flg,ierr))

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
  PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_WRITE,view_out,ierr))
  PetscCallA(PetscViewerBinaryGetDescriptor(view_out,fd,ierr))

  ! Write binary output
  PetscCallA(PetscBinaryWrite(fd,t,one,PETSC_INT,ierr))
  PetscCallA(PetscBinaryWrite(fd,array,m,PETSC_SCALAR,ierr))

  ! Destroy the output viewer and work array
  PetscCallA(PetscViewerDestroy(view_out,ierr))
  deallocate(array)

  ! ----------------------------------------------------------------------
  !          PART 2: Read data from file and form a vector
  ! ----------------------------------------------------------------------

  ! Open input binary viewer
  PetscCallA(PetscViewerBinaryOpen(PETSC_COMM_SELF,"input.dat",FILE_MODE_READ,view_in,ierr))
  PetscCallA(PetscViewerBinaryGetDescriptor(view_in,fd,ierr))

  ! Create vector and get pointer to data space
  PetscCallA(VecCreate(PETSC_COMM_SELF,vec,ierr))
  PetscCallA(VecSetSizes(vec,PETSC_DECIDE,m,ierr))

  PetscCallA(VecSetFromOptions(vec,ierr))

  PetscCallA(VecGetArrayF90(vec,avec,ierr))

  ! Read data into vector
  PetscCallA(PetscBinaryRead(fd,t,one,PETSC_NULL_INTEGER,PETSC_INT,ierr))
  sz=t(1)

  if (sz <= 0) then
   SETERRA(PETSC_COMM_SELF,PETSC_ERR_USER,"Error: Must have array length > 0")
  endif

  write(outstring,'(a,i2.2,a)') "reading data in binary from input.dat, sz =", sz, " ...\n"
  PetscCallA(PetscPrintf(PETSC_COMM_SELF,trim(outstring),ierr))

  PetscCallA(PetscBinaryRead(fd,avec,sz,PETSC_NULL_INTEGER,PETSC_SCALAR,ierr))

  ! View vector
  PetscCallA(VecRestoreArrayF90(vec,avec,ierr))
  PetscCallA(VecView(vec,PETSC_VIEWER_STDOUT_SELF,ierr))

  ! Free data structures
  deallocate(t)
  PetscCallA(VecDestroy(vec,ierr))
  PetscCallA(PetscViewerDestroy(view_in,ierr))
  PetscCallA(PetscFinalize(ierr))

  end program

!/*TEST
!
!     test:
!        output_file: output/ex6_1.out
!
!TEST*/
