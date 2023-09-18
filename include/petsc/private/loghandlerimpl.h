#pragma once

#include <petsc/private/petscimpl.h>

typedef struct _PetscLogHandlerOps *PetscLogHandlerOps;
struct _PetscLogHandlerOps {
  PetscErrorCode (*destroy)(PetscLogHandler);
  PetscErrorCode (*eventbegin)(PetscLogHandler, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
  PetscErrorCode (*eventend)(PetscLogHandler, PetscLogEvent, PetscObject, PetscObject, PetscObject, PetscObject);
  PetscErrorCode (*eventsync)(PetscLogHandler, PetscLogEvent, MPI_Comm);
  PetscErrorCode (*objectcreate)(PetscLogHandler, PetscObject);
  PetscErrorCode (*objectdestroy)(PetscLogHandler, PetscObject);
  PetscErrorCode (*stagepush)(PetscLogHandler, PetscLogStage);
  PetscErrorCode (*stagepop)(PetscLogHandler, PetscLogStage);
  PetscErrorCode (*view)(PetscLogHandler, PetscViewer);
};

struct _p_PetscLogHandler {
  PETSCHEADER(struct _PetscLogHandlerOps);
  PetscLogState state;
  void         *data;
};

PETSC_INTERN PetscErrorCode PetscLogHandlerPackageInitialize(void);
PETSC_INTERN PetscErrorCode PetscLogHandlerLogObjectState_Internal(PetscLogHandler, PetscObject, const char *, va_list);
