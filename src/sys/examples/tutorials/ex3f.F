!
!    "$Id: ex3f.F,v 1.15 2001/01/15 21:44:21 bsmith Exp $";
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
!      - Note that PetscLogEventRegister() returns to the user a unique
!        integer event number, which should then be used for profiling
!        the event via PetscLogEventBegin() and PetscLogEventEnd().
!      - The user can also optionally log floating point operations
!        with the routine PetscLogFlops().
!
      call PetscLogEventRegister(USER_EVENT1,'Event 1',0,ierr)
      call PetscLogEventRegister(USER_EVENT2,'Event 2',0,ierr)
      call PetscLogEventRegister(USER_EVENT3,'Event 3',0,ierr)
      call PetscLogEventRegister(USER_EVENT4,'Event 4',0,ierr)
      call PetscLogEventRegister(USER_EVENT5,'Event 5',0,ierr)
      call PetscLogEventRegister(USER_EVENT6,'Event 6',0,ierr)
      call PetscLogEventRegister(USER_EVENT7,'Event 7',0,ierr)
      call PetscLogEventRegister(USER_EVENT8,'Event 8',0,ierr)
      call PetscLogEventRegister(USER_EVENT9,'Event 9',0,ierr)
      call PetscLogEventBegin(USER_EVENT1,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT1,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogEventBegin(USER_EVENT2,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT2,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogEventBegin(USER_EVENT3,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT3,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogEventBegin(USER_EVENT4,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT4,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     & PETSC_NULL_OBJECT,ierr)
      call PetscLogEventBegin(USER_EVENT5,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT5,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogEventBegin(USER_EVENT6,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT6,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogEventBegin(USER_EVENT7,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT7,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogEventBegin(USER_EVENT8,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT8,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogEventBegin(USER_EVENT9,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscLogFlops(imax,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT9,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
!
!    We disable the logging of an event.
!      - Note that activation/deactivation of PETSc events and MPE 
!        events is handled separately.
!      - Note that the user can activate/deactive both user-defined
!        events and predefined PETSc events.
!
      call PetscLogEventMPEDeactivate(USER_EVENT1,ierr)
      call PetscLogEventDeactivate(USER_EVENT1,ierr)
      call PetscLogEventBegin(USER_EVENT1,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT1,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
!
!    We next enable the logging of an event
!
      call PetscLogEventMPEActivate(USER_EVENT1,ierr)
      call PetscLogEventActivate(USER_EVENT1,ierr)
      call PetscLogEventBegin(USER_EVENT1,PETSC_NULL_OBJECT,                &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)
      call PetscSleep(1,ierr)
      call PetscLogEventEnd(USER_EVENT1,PETSC_NULL_OBJECT,                  &
     &     PETSC_NULL_OBJECT,PETSC_NULL_OBJECT,                         &
     &     PETSC_NULL_OBJECT,ierr)

      call PetscFinalize(ierr)

      end
