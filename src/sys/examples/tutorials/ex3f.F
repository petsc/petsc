!
!
!   Description: Demonstrates how users can augment the PETSc profiling by
!                nserting their own event logging.
!
!/*T
!   Concepts: PetscLog^user-defined event profiling (basic example);
!   Concepts: PetscLog^activating/deactivating events for profiling (basic example);
!   Processors: n
!T*/
! -----------------------------------------------------------------------

      program main
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  The following include statements are required for using PetscLog Routines
!
#include "finclude/petscsys.h"
#include "finclude/petsclog.h"
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
      integer imax
      PetscErrorCode ierr
      parameter (imax = 10000)
      PetscLogDouble onefp
      parameter (onefp = 1.0d0)
      PetscReal onereal,tenreal
      parameter (onereal = 1.0, tenreal = 10.0)
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

!
!     Create a new user-defined event.
!      - Note that PetscLogEventRegister() returns to the user a unique
!        integer event number, which should then be used for profiling
!        the event via PetscLogEventBegin() and PetscLogEventEnd().
!      - The user can also optionally log floating point operations
!        with the routine PetscLogFlops().
!
      call PetscLogEventRegister('Event 1',0,USER_EVENT1,ierr)
      call PetscLogEventRegister('Event 2',0,USER_EVENT2,ierr)
      call PetscLogEventRegister('Event 3',0,USER_EVENT3,ierr)
      call PetscLogEventRegister('Event 4',0,USER_EVENT4,ierr)
      call PetscLogEventRegister('Event 5',0,USER_EVENT5,ierr)
      call PetscLogEventRegister('Event 6',0,USER_EVENT6,ierr)
      call PetscLogEventRegister('Event 7',0,USER_EVENT7,ierr)
      call PetscLogEventRegister('Event 8',0,USER_EVENT8,ierr)
      call PetscLogEventRegister('Event 9',0,USER_EVENT9,ierr)
      call PetscLogEventBegin(USER_EVENT1,ierr)
      call PetscLogFlops(imax*onefp,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT1,ierr)
      call PetscLogEventBegin(USER_EVENT2,ierr)
      call PetscLogFlops(imax*onefp,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT2,ierr)
      call PetscLogEventBegin(USER_EVENT3,ierr)
      call PetscLogFlops(imax*onefp,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT3,ierr)
      call PetscLogEventBegin(USER_EVENT4,ierr)
      call PetscLogFlops(imax*onefp,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT4,ierr)
      call PetscLogEventBegin(USER_EVENT5,ierr)
      call PetscLogFlops(imax*onefp,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT5,ierr)
      call PetscLogEventBegin(USER_EVENT6,ierr)
      call PetscLogFlops(imax*onefp,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT6,ierr)
      call PetscLogEventBegin(USER_EVENT7,ierr)
      call PetscLogFlops(imax*onefp,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT7,ierr)
      call PetscLogEventBegin(USER_EVENT8,ierr)
      call PetscLogFlops(imax*onefp,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT8,ierr)
      call PetscLogEventBegin(USER_EVENT9,ierr)
      call PetscLogFlops(imax*onefp,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT9,ierr)
!
!    We disable the logging of an event.
!      - Note that the user can activate/deactive both user-defined
!        events and predefined PETSc events.
!
      call PetscLogEventDeactivate(USER_EVENT1,ierr)
      call PetscLogEventBegin(USER_EVENT1,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT1,ierr)
!
!    We next enable the logging of an event
!
      call PetscLogEventActivate(USER_EVENT1,ierr)
      call PetscLogEventBegin(USER_EVENT1,ierr)
      call PetscSleep(onereal,ierr)
      call PetscLogEventEnd(USER_EVENT1,ierr)

      call PetscInfo('PETSc info message\n'//'Another line\n',ierr)
      call PetscFinalize(ierr)

      end

