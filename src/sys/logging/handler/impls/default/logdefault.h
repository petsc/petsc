#if !defined(_PETSCLOGDEFAULT_H)
  #define _PETSCLOGDEFAULT_H

  #include <petsc/private/loghandlerimpl.h> /*I "petscsys.h" I*/
  #include <petsc/private/logimpl.h>        /*I "petscsys.h" I*/

PETSC_INTERN PetscErrorCode PetscLogHandlerCreate_Default(PetscLogHandler);

PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetEventPerfInfo(PetscLogHandler, PetscLogStage, PetscLogEvent, PetscEventPerfInfo **);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetLogActions(PetscLogHandler, PetscBool);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultSetLogObjects(PetscLogHandler, PetscBool);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultLogObjectState(PetscLogHandler, PetscObject, const char[], va_list);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultGetNumObjects(PetscLogHandler, PetscInt *);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultDeactivatePush(PetscLogHandler, PetscLogStage, PetscLogEvent);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultDeactivatePop(PetscLogHandler, PetscLogStage, PetscLogEvent);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultEventsPause(PetscLogHandler);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultEventsResume(PetscLogHandler);
PETSC_INTERN PetscErrorCode PetscLogHandlerDump_Default(PetscLogHandler, const char[]);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultStageSetVisible(PetscLogHandler, PetscLogStage, PetscBool);
PETSC_INTERN PetscErrorCode PetscLogHandlerDefaultStageGetVisible(PetscLogHandler, PetscLogStage, PetscBool *);
#endif // #define _PETSCLOGDEFAULT_H
