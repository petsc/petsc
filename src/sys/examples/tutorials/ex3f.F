!
!    "$Id: ex3f.F,v 1.11 2000/09/06 22:19:13 balay Exp bsmith $";
!
!   Description: Demonstrates how users can augment the PETSc profiling by
!                nserting their own event logging.
!
!/*T
!   Concepts: PLog^User-defined event profiling (basic example);
!   Concepts: PLog^Activating/deactivating events for profiling (basic example);
!   Routines: PLogEventRegister(); PLogEventBegin(); PLogEventEnd();
!   Routines: PLogEventDeactivate(); PLogEventActivate(); PLogFlops();
!   Routines: PLogEventMPEDeactivate(); PLogEventMPEActivate();
!   Routines: PetscSleep();
!   Processors: n
!T*/
! -----------------------------------------------------------------------

      program main
      implicit none

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                    Include files
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!  The following include statements are required for using PLog Routines
!
#include "include/finclude/petsc.h"
#include "include/finclude/petsclog.h"
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                   Variable declarations
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
      integer USER_EVENT1,USER_EVENT2,USER_EVENT3,USER_EVENT4
      integer USER_EVENT8,USER_EVENT9
      integer USER_EVENT5,USER_EVENT6,USER_EVENT7,imax,ierr
      parameter (imax = 10000)
!
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!                 Beginning of program
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

!
!     Create a new user-defined event.
!      - Note that PLogEventRegister() returns to the user a unique
!        integer event number, which should then be used for profiling
!        the event via PLogEventBegin() and PLogEventEnd().
!      - The user can also optionally log floating point operations
!        with the routine PLogFlops().
!
      call PLogEventRegister(USER_EVENT1,'User event 1','Red:',ierr)
      call PLogEventRegister(USER_EVENT2,'User event 2','Red:',ierr)
      call PLogEventRegister(USER_EVENT3,'User event 3','Red:',ierr)
      call PLogEventRegister(USER_EVENT4,'User event 4','Red:',ierr)
      call PLogEventRegister(USER_EVENT5,'User event 5','Red:',ierr)
      call PLogEventRegister(USER_EVENT6,'User event 6','Red:',ierr)
      call PLogEventRegister(USER_EVENT7,'User event 7','Red:',ierr)
      call PLogEventRegister(USER_EVENT8,'User event 8','Red:',ierr)
      call PLogEventRegister(USER_EVENT9,'User event 9','Red:',ierr)
      call PLogEventBegin(USER_EVENT1,PETSC_NULL_OBJECT,                                           &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT1,PETSC_NULL_OBJECT,                            &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogEventBegin(USER_EVENT2,PETSC_NULL_OBJECT,                          &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT2,PETSC_NULL_OBJECT,                            &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogEventBegin(USER_EVENT3,PETSC_NULL_OBJECT,                          &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT3,PETSC_NULL_OBJECT,                            &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogEventBegin(USER_EVENT4,PETSC_NULL_OBJECT,                          &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT4,PETSC_NULL_OBJECT,                            &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogEventBegin(USER_EVENT5,PETSC_NULL_OBJECT,                          &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT5,PETSC_NULL_OBJECT,                            &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogEventBegin(USER_EVENT6,PETSC_NULL_OBJECT,                          &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT6,PETSC_NULL_OBJECT,                            &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogEventBegin(USER_EVENT7,PETSC_NULL_OBJECT,                          &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT7,PETSC_NULL_OBJECT,                            &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogEventBegin(USER_EVENT8,PETSC_NULL_OBJECT,                          &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT8,PETSC_NULL_OBJECT,                            &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogEventBegin(USER_EVENT9,PETSC_NULL_OBJECT,                          &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT9,PETSC_NULL_OBJECT,                            &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
!
!    We disable the logging of an event.
!      - Note that activation/deactivation of PETSc events and MPE 
!        events is handled separately.
!      - Note that the user can activate/deactive both user-defined
!        events and predefined PETSc events.
!
      call PLogEventMPEDeactivate(USER_EVENT1,ierr)
      call PLogEventDeactivate(USER_EVENT1,ierr)
      call PLogEventBegin(USER_EVENT1,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT1,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
!
!    We next enable the logging of an event
!
      call PLogEventMPEActivate(USER_EVENT1,ierr)
      call PLogEventActivate(USER_EVENT1,ierr)
      call PLogEventBegin(USER_EVENT1,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)
      call PetscSleep(1,ierr)
      call PLogEventEnd(USER_EVENT1,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,ierr)

      call PetscFinalize(ierr)

      end
