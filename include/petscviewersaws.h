#pragma once

#include <petscviewer.h>
#include <SAWs.h>

/* MANSEC = Viewer */

PETSC_EXTERN PetscErrorCode PetscViewerSAWsOpen(MPI_Comm, PetscViewer *);
PETSC_EXTERN PetscViewer    PETSC_VIEWER_SAWS_(MPI_Comm);
#define PETSC_VIEWER_SAWS_WORLD PETSC_VIEWER_SAWS_(PETSC_COMM_WORLD)
#define PETSC_VIEWER_SAWS_SELF  PETSC_VIEWER_SAWS_(PETSC_COMM_SELF)

/*MC
  PetscCallSAWs - Calls a SAWs function and then checks the resulting error code, if it is
  non-zero it calls the error handler and returns from the current function with the error
  code `PETSC_ERR_LIB`.

  Synopsis:
  #include <petscviewersaws.h>
  void PetscCallSAWs(func, args)

  Not Collective

  Input Parameters:
+ func - any SAWs function that returns an error code
- args - the arguments to the function

  Level: beginner

.seealso: `PetscCall()`, `SETERRQ()`, `PetscCheck()`, `PetscAssert()`, `PetscTraceBackErrorHandler()`, `PetscCallMPI()`,
          `PetscPushErrorHandler()`, `PetscError()`, `CHKMEMQ`, `CHKERRA()`,
          `CHKERRMPI()`, `PetscCallBack()`, `PetscCallAbort()`, `PetscCallVoid()`, `PetscCallNull()`
M*/
#define PetscCallSAWs(func, args) \
  do { \
    int _ierr; \
    PetscStackPushExternal(#func); \
    _ierr = func args; \
    PetscStackPop; \
    PetscCheck(!_ierr, PETSC_COMM_SELF, PETSC_ERR_LIB, "Error in %s() %d", #func, _ierr); \
  } while (0)
