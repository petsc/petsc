!
!
!   Description: Demonstrates how users can augment the PETSc profiling by
!                inserting their own event logging.
!
! -----------------------------------------------------------------------

      program SchoolDay
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petsclog.h>
      use petscmpi  ! or mpi or mpi_f08
      use petscsys
      implicit none

!====================================================================
!     Local Variables

      ! Settings:
      integer, parameter        :: verbose=0               ! 0: silent, >=1 : increasing amount of debugging output
      integer, parameter        :: msgLen = 30             ! number of reals which is sent with MPI_Isend
      PetscReal, parameter      :: second=0.1;             ! time is sped up by a factor 10

      ! Codes
      integer, parameter        :: BOY=1, GIRL=2, TEACHER=0
      PetscMPIInt, parameter    :: tagMsg   = 1200;

      ! Timers
      PetscLogEvent :: Morning,  Afternoon
      PetscLogEvent :: PlayBall, SkipRope
      PetscLogEvent :: TidyClass
      PetscLogEvent :: Lessons,  CorrectHomework
      PetscClassId classid

      ! Petsc-stuff
      PetscErrorCode            :: ierr

      ! MPI-stuff
      PetscMPIInt              :: rank, size
      PetscReal, allocatable    :: message(:,:)
      integer                   :: item, maxItem
      integer4                  :: status(MPI_STATUS_SIZE)
      PetscMPIInt                  req

      ! Own stuff
      integer4                  :: role                 ! is this process a BOY, a GIRL or a TEACHER?
      integer4                  :: i, j
      integer4,parameter        :: one=1
!====================================================================
!     Initializations
      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)
      if (ierr .ne. 0) then
        print*,'Unable to initialize PETSc'
        stop
      endif
      call MPI_Comm_size(PETSC_COMM_WORLD, size,ierr)
      call MPI_Comm_rank(PETSC_COMM_WORLD, rank,ierr)

      if (rank==0) then
         role = TEACHER
      else if (rank<0.4*size) then
         role = GIRL
      else
         role = BOY
      end if

      allocate(message(msgLen,msglen))
      do i = 1,msgLen
         do j  = 1,msgLen
            message(i,j) = 10.0*j + i*1.0/(rank+one)
         end do
      end do
!
!====================================================================
!     Create new user-defined events
      classid = 0
      call PetscLogEventRegister('Morning',         classid, Morning,   ierr)
      call PetscLogEventRegister('Afternoon',       classid, Afternoon, ierr)
      call PetscLogEventRegister('Play Ball',       classid, PlayBall,  ierr)
      call PetscLogEventRegister('Skip Rope',       classid, SkipRope,  ierr)
      call PetscLogEventRegister('Tidy Classroom',  classid, TidyClass, ierr)
      call PetscLogEventRegister('Lessons',         classid, Lessons,   ierr)
      call PetscLogEventRegister('Correct Homework',classid,CorrectHomework,          &
     &                                                            ierr)
      if (verbose>=1) then
      print '(a,i0,a)','[',rank,'] SchoolDay events have been defined'
      endif

!====================================================================
!     Go through the school day
      call PetscLogEventBegin(Morning,ierr)

         call PetscLogFlops(190000d0,ierr)
         call PetscSleep(0.5*second,ierr)

         call PetscLogEventBegin(Lessons,ierr)
         call PetscLogFlops(23000d0,ierr)
         call PetscSleep(1*second, ierr)
         if (size>1) then
         call MPI_Isend( message, msgLen, MPI_DOUBLE_PRECISION,                             &
     &                        mod(rank+1,size),                                             &
     &                        tagMsg+rank, PETSC_COMM_WORLD, req, ierr)
         call  MPI_Recv( message, msgLen, MPI_DOUBLE_PRECISION,                             &
     &                       mod(rank-1+size,size),                                         &
     &                  tagMsg+mod(rank-1+size,size), PETSC_COMM_WORLD,                     &
     &        status, ierr)
         call MPI_Wait(req,MPI_STATUS_IGNORE,ierr)
         end if
         call PetscLogEventEnd(Lessons,ierr)

         if (role==TEACHER) then
            call PetscLogEventBegin(TidyClass,ierr)
            call PetscLogFlops(600000d0,ierr)
            call PetscSleep(0.6*second, ierr)
               call PetscLogEventBegin(CorrectHomework,ierr)
               call PetscLogFlops(234700d0,ierr)
               call PetscSleep(0.4*second, ierr)
               call PetscLogEventEnd(CorrectHomework,ierr)
            call PetscLogEventEnd(TidyClass,ierr)
         else if (role==BOY) then
            call PetscLogEventBegin(SkipRope,ierr)
            call PetscSleep(0.8*second, ierr)
            call PetscLogEventEnd(SkipRope,ierr)
         else
            call PetscLogEventBegin(PlayBall,ierr)
            call PetscSleep(0.9*second, ierr)
            call PetscLogEventEnd(PlayBall,ierr)
         end if

         call PetscLogEventBegin(Lessons,ierr)
         call PetscLogFlops(120000d0,ierr)
         call PetscSleep(0.7*second, ierr)
         call PetscLogEventEnd(Lessons,ierr)

      call PetscLogEventEnd(Morning,ierr)

      call PetscLogEventBegin(Afternoon,ierr)

         item = rank*(3-rank)
         call MPI_Allreduce(item, maxItem, 1, MPI_INTEGER, MPI_MAX,                    &
     &                           PETSC_COMM_WORLD, ierr)

         item = rank*(10-rank)
         call MPI_Allreduce(item, maxItem, 1, MPI_INTEGER, MPI_MAX,                    &
     &                           PETSC_COMM_WORLD, ierr)

         call PetscLogFlops(58988d0,ierr)
         call PetscSleep(0.6*second,ierr)

         call PetscLogEventBegin(Lessons,ierr)
         call PetscLogFlops(123456d0,ierr)
         call PetscSleep(1*second, ierr)
         call PetscLogEventEnd(Lessons,ierr)

         if (role==TEACHER) then
            call PetscLogEventBegin(TidyClass,ierr)
            call PetscLogFlops(17800d0,ierr)
            call PetscSleep(1.1*second, ierr)
            call PetscLogEventBegin(Lessons,ierr)
            call PetscLogFlops(72344d0,ierr)
            call PetscSleep(0.5*second, ierr)
            call PetscLogEventEnd(Lessons,ierr)
            call PetscLogEventEnd(TidyClass,ierr)
         else if (role==GIRL) then
            call PetscLogEventBegin(SkipRope,ierr)
            call PetscSleep(0.7*second, ierr)
            call PetscLogEventEnd(SkipRope,ierr)
         else
            call PetscLogEventBegin(PlayBall,ierr)
            call PetscSleep(0.8*second, ierr)
            call PetscLogEventEnd(PlayBall,ierr)
         end if

         call PetscLogEventBegin(Lessons,ierr)
         call PetscLogFlops(72344d0,ierr)
         call PetscSleep(0.5*second, ierr)
         call PetscLogEventEnd(Lessons,ierr)

      call PetscLogEventEnd(Afternoon,ierr)

      if (.false.) then
         continue
      else if (role==TEACHER) then
         call PetscLogEventBegin(TidyClass,ierr)
         call PetscLogFlops(612300d0,ierr)
         call PetscSleep(1.1*second, ierr)
         call PetscLogEventEnd(TidyClass,ierr)
         call PetscLogEventBegin(CorrectHomework,ierr)
         call PetscLogFlops(234700d0,ierr)
         call PetscSleep(1.1*second, ierr)
         call PetscLogEventEnd(CorrectHomework,ierr)
      else
         call PetscLogEventBegin(SkipRope,ierr)
         call PetscSleep(0.7*second, ierr)
         call PetscLogEventEnd(SkipRope,ierr)
         call PetscLogEventBegin(PlayBall,ierr)
         call PetscSleep(0.8*second, ierr)
         call PetscLogEventEnd(PlayBall,ierr)
      end if

      call PetscLogEventBegin(Lessons,ierr)
      call PetscLogFlops(120000d0,ierr)
      call PetscSleep(0.7*second, ierr)
      call PetscLogEventEnd(Lessons,ierr)

      call PetscSleep(0.25*second,ierr)

      call PetscLogEventBegin(Morning,ierr)

         call PetscLogFlops(190000d0,ierr)
         call PetscSleep(0.5*second,ierr)

         call PetscLogEventBegin(Lessons,ierr)
         call PetscLogFlops(23000d0,ierr)
         call PetscSleep(1*second, ierr)
         if (size>1) then
         call MPI_Isend( message, msgLen, MPI_DOUBLE_PRECISION,                             &
     &                        mod(rank+1,size),                                             &
     &                   tagMsg+rank, PETSC_COMM_WORLD, req, ierr)
         call MPI_Recv( message, msgLen, MPI_DOUBLE_PRECISION,                              &
     &                  mod(rank-1+size,size),                                              &
     &                  tagMsg+mod(rank-1+size,size), PETSC_COMM_WORLD,                     &
     &                   status, ierr)
         call MPI_Wait(req,MPI_STATUS_IGNORE,ierr)
         end if
         call PetscLogEventEnd(Lessons,ierr)

         if (role==TEACHER) then
            call PetscLogEventBegin(TidyClass,ierr)
            call PetscLogFlops(600000d0,ierr)
            call PetscSleep(1.2*second, ierr)
            call PetscLogEventEnd(TidyClass,ierr)
         else if (role==BOY) then
            call PetscLogEventBegin(SkipRope,ierr)
            call PetscSleep(0.8*second, ierr)
            call PetscLogEventEnd(SkipRope,ierr)
         else
            call PetscLogEventBegin(PlayBall,ierr)
            call PetscSleep(0.9*second, ierr)
            call PetscLogEventEnd(PlayBall,ierr)
         end if

         call PetscLogEventBegin(Lessons,ierr)
         call PetscLogFlops(120000d0,ierr)
         call PetscSleep(0.7*second, ierr)
         call PetscLogEventEnd(Lessons,ierr)

      call PetscLogEventEnd(Morning,ierr)

      deallocate(message)

      call PetscFinalize(ierr)

      end program SchoolDay

!/*TEST
!
! testset:
!   suffix: no_log
!   requires: !defined(PETSC_USE_LOG)
!   test:
!     suffix: ascii
!     args: -log_view ascii:filename.txt
!   test:
!     suffix: detail
!     args: -log_view ascii:filename.txt:ascii_info_detail
!   test:
!     suffix: xml
!     args: -log_view ascii:filename.xml:ascii_xml
!
! testset:
!   args: -log_view ascii:filename.txt
!   output_file: output/ex3f90.out
!   requires: defined(PETSC_USE_LOG)
!   test:
!     suffix: 1
!     nsize: 1
!   test:
!     suffix: 2
!     nsize: 2
!   test:
!     suffix: 3
!     nsize: 3
!
! testset:
!   suffix: detail
!   args: -log_view ascii:filename.txt:ascii_info_detail
!   output_file: output/ex3f90.out
!   requires: defined(PETSC_USE_LOG)
!   test:
!     suffix: 1
!     nsize: 1
!   test:
!     suffix: 2
!     nsize: 2
!   test:
!     suffix: 3
!     nsize: 3
!
! testset:
!   suffix: xml
!   args: -log_view ascii:filename.xml:ascii_xml
!   output_file: output/ex3f90.out
!   requires: defined(PETSC_USE_LOG)
!   test:
!     suffix: 1
!     nsize: 1
!   test:
!     suffix: 2
!     nsize: 2
!   test:
!     suffix: 3
!     nsize: 3
!
!TEST*/
