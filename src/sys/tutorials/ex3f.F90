!
!
!   Description: Demonstrates how users can augment the PETSc profiling by
!                nserting their own event logging.
!
! -----------------------------------------------------------------------

      program main
#include <petsc/finclude/petscsys.h>
#include <petsc/finclude/petsclog.h>
      use petscsys
      implicit none

!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
      PetscLogEvent USER_EVENT1,USER_EVENT2
      PetscLogEvent USER_EVENT3,USER_EVENT4
      PetscLogEvent USER_EVENT5,USER_EVENT6
      PetscLogEvent USER_EVENT7,USER_EVENT8
      PetscLogEvent USER_EVENT9
      PetscClassId  classid
      integer imax
      PetscErrorCode ierr
      parameter (imax = 10000)
      PetscLogDouble onefp
      parameter (onefp = 1.0d0)
      PetscReal onereal,tenreal
      parameter (onereal = 1.0, tenreal = 10.0)
      PetscInt n
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

      PetscCallA(PetscInitialize(PETSC_NULL_CHARACTER,ierr))

!
!     Create a new user-defined event.
!      - Note that PetscLogEventRegister() returns to the user a unique
!        integer event number, which should then be used for profiling
!        the event via PetscLogEventBegin() and PetscLogEventEnd().
!      - The user can also optionally log floating point operations
!        with the routine PetscLogFlops().
!
      classid = 0
      PetscCallA(PetscLogEventRegister('Event 1',classid,USER_EVENT1,ierr))
      PetscCallA(PetscLogEventRegister('Event 2',classid,USER_EVENT2,ierr))
      PetscCallA(PetscLogEventRegister('Event 3',classid,USER_EVENT3,ierr))
      PetscCallA(PetscLogEventRegister('Event 4',classid,USER_EVENT4,ierr))
      PetscCallA(PetscLogEventRegister('Event 5',classid,USER_EVENT5,ierr))
      PetscCallA(PetscLogEventRegister('Event 6',classid,USER_EVENT6,ierr))
      PetscCallA(PetscLogEventRegister('Event 7',classid,USER_EVENT7,ierr))
      PetscCallA(PetscLogEventRegister('Event 8',classid,USER_EVENT8,ierr))
      PetscCallA(PetscLogEventRegister('Event 9',classid,USER_EVENT9,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT1,ierr))
      PetscCallA(PetscLogFlops(imax*onefp,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT1,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT2,ierr))
      PetscCallA(PetscLogFlops(imax*onefp,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT2,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT3,ierr))
      PetscCallA(PetscLogFlops(imax*onefp,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT3,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT4,ierr))
      PetscCallA(PetscLogFlops(imax*onefp,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT4,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT5,ierr))
      PetscCallA(PetscLogFlops(imax*onefp,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT5,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT6,ierr))
      PetscCallA(PetscLogFlops(imax*onefp,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT6,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT7,ierr))
      PetscCallA(PetscLogFlops(imax*onefp,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT7,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT8,ierr))
      PetscCallA(PetscLogFlops(imax*onefp,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT8,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT9,ierr))
      PetscCallA(PetscLogFlops(imax*onefp,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT9,ierr))
!
!    We disable the logging of an event.
!      - Note that the user can activate/deactive both user-defined
!        events and predefined PETSc events.
!
      PetscCallA(PetscLogEventDeactivate(USER_EVENT1,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT1,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT1,ierr))
!
!    We next enable the logging of an event
!
      PetscCallA(PetscLogEventActivate(USER_EVENT1,ierr))
      PetscCallA(PetscLogEventBegin(USER_EVENT1,ierr))
      PetscCallA(PetscSleep(onereal,ierr))
      PetscCallA(PetscLogEventEnd(USER_EVENT1,ierr))

      PetscCallA(PetscInfo('PETSc info message\n'//'Another line\n',ierr))
      PetscCallA(PetscOptionsAllUsed(PETSC_NULL_OPTIONS,n,ierr));
      PetscCallA(PetscFinalize(ierr))

      end

!
!/*TEST
!
!   test:
!
!TEST*/
