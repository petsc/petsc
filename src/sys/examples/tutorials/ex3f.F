

C    "$Id: ex1f.F,v 1.1 1996/10/10 23:58:02 balay Exp balay $";
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
#include "include/FINCLUDE/plog.h"
C
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C                   Variable declarations
C - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
C
      integer USER_EVENT,imax,ierr
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
      call PLogEventRegister(USER_EVENT,'User event      ','Red:',ierr)
      call PLogEventBegin(USER_EVENT,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PLogFlops(imax)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
C
C    We disable the logging of an event.
C      - Note that activation/deactivation of PETSc events and MPE 
C        events is handled separately.
C      - Note that the user can activate/deactive both user-defined
C        events and predefined PETSc events.
C
      call PLogEventMPEDeactivate(USER_EVENT,ierr)
      call PLogEventDeactivate(USER_EVENT,ierr)
      call PLogEventBegin(USER_EVENT,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
C
C    We next enable the logging of an event
C
      call PLogEventMPEActivate(USER_EVENT,ierr)
      call PLogEventActivate(USER_EVENT,ierr)
      call PLogEventBegin(USER_EVENT,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)
      call PetscSleep(1)
      call PLogEventEnd(USER_EVENT,PETSC_NULL,PETSC_NULL,PETSC_NULL,
     * PETSC_NULL)

      call PetscFinalize(ierr)

      end
