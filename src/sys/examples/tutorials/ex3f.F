

C    "$Id: ex1f.F,v 1.3 1996/12/09 15:54:18 balay Exp bsmith $";
C
C   Description: Demonstrates how users can augment the PETSc profiling by
C                nserting their own event logging.
C
C/*T
C   Concepts: PLog^User-defined event profiling (basic example);
C   Concepts: PLog^Activating/deactivating events for profiling (basic example);
C   Routines: PLogEventRegister(); PLogEventBegin(); PLogEventEnd();
C   Routines: PLogEventDeactivate(); PLogEventActivate(); PLogFlops();
C   Routines: PLogEventMPEDeactivate(); PLogEventMPEActivate();
C   Routines: PetscSleep();
C   Processors: n
CT*/
C -----------------------------------------------------------------------

      program main
      implicit none

C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                    Include files
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
C  The following include statements are required for using PLog Routines
C
#include "include/FINCLUDE/petsc.h"
#include "include/FINCLUDE/petsclog.h"
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                   Variable declarations
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
      integer USER_EVENT1,USER_EVENT2,USER_EVENT3,USER_EVENT4
      integer USER_EVENT8,USER_EVENT9
      integer USER_EVENT5,USER_EVENT6,USER_EVENT7,imax,ierr
      parameter (imax = 10000)
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                 Beginning of program
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

      call PetscInitialize(PETSC_NULL_CHARACTER,ierr)

C
C     Create a new user-defined event.
C      - Note that PLogEventRegister() returns to the user a unique
C        integer event number, which should then be used for profiling
C        the event via PLogEventBegin() and PLogEventEnd().
C      - The user can also optionally log floating point operations
C        with the routine PLogFlops().
C
      call PLogEventRegister(USER_EVENT1,'User event 1   ','Red:',ierr)
      call PLogEventRegister(USER_EVENT2,'User event 2   ','Red:',ierr)
      call PLogEventRegister(USER_EVENT3,'User event 3   ','Red:',ierr)
      call PLogEventRegister(USER_EVENT4,'User event 4   ','Red:',ierr)
      call PLogEventRegister(USER_EVENT5,'User event 5   ','Red:',ierr)
      call PLogEventRegister(USER_EVENT6,'User event 6   ','Red:',ierr)
      call PLogEventRegister(USER_EVENT7,'User event 7   ','Red:',ierr)
      call PLogEventRegister(USER_EVENT8,'User event 8   ','Red:',ierr)
      call PLogEventRegister(USER_EVENT9,'User event 9   ','Red:',ierr)
      call PLogEventBegin(USER_EVENT1,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogFlops(imax)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT1,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogEventBegin(USER_EVENT2,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogFlops(imax)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT2,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogEventBegin(USER_EVENT3,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogFlops(imax)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT3,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogEventBegin(USER_EVENT4,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogFlops(imax)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT4,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogEventBegin(USER_EVENT5,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogFlops(imax)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT5,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogEventBegin(USER_EVENT6,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogFlops(imax)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT6,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogEventBegin(USER_EVENT7,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogFlops(imax)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT7,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogEventBegin(USER_EVENT8,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogFlops(imax)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT8,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogEventBegin(USER_EVENT9,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogFlops(imax)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT9,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
C
C    We disable the logging of an event.
C      - Note that activation/deactivation of PETSc events and MPE 
C        events is handled separately.
C      - Note that the user can activate/deactive both user-defined
C        events and predefined PETSc events.
C
      call PLogEventMPEDeactivate(USER_EVENT1,ierr)
      call PLogEventDeactivate(USER_EVENT1,ierr)
      call PLogEventBegin(USER_EVENT1,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT1,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
C
C    We next enable the logging of an event
C
      call PLogEventMPEActivate(USER_EVENT1,ierr)
      call PLogEventActivate(USER_EVENT1,ierr)
      call PLogEventBegin(USER_EVENT1,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT1,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)

      call PetscFinalize(ierr)

      end
