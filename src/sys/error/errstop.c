#include <petscsys.h> /*I "petscsys.h" I*/
#include "err.h"

/*@C
  PetscMPIAbortErrorHandler - Calls `PETSCABORT()` and exits.

  Not Collective, No Fortran Support

  Input Parameters:
+ comm - communicator over which error occurred
. line - the line number of the error (indicated by `__LINE__`)
. fun  - the function name
. file - the file in which the error was detected (indicated by `__FILE__`)
. mess - an error text string, usually just printed to the screen
. n    - the generic error number
. p    - `PETSC_ERROR_INITIAL` if error just detected, otherwise `PETSC_ERROR_REPEAT`
- ctx  - error handler context

  Level: developer

  Note:
  Users do not directly call this routine

  Use `PetscPushErrorHandler()` to set the desired error handler.  The
  currently available PETSc error handlers include `PetscTraceBackErrorHandler()`,
  `PetscMPIAbortErrorHandler()`, `PetscAttachDebuggerErrorHandler()`, and `PetscAbortErrorHandler()`.

.seealso: `PetscError()`, `PetscPushErrorHandler()`, `PetscPopErrorHandler()`, `PetscAttachDebuggerErrorHandler()`,
          `PetscAbortErrorHandler()`, `PetscTraceBackErrorHandler()`, `PetscEmacsClientErrorHandler()`, `PetscReturnErrorHandler()`
 @*/
PetscErrorCode PetscMPIAbortErrorHandler(MPI_Comm comm, int line, const char *fun, const char *file, PetscErrorCode n, PetscErrorType p, const char *mess, void *ctx)
{
  PetscFunctionBegin;
  if (!mess) mess = " ";

  if (n == PETSC_ERR_MEM || n == PETSC_ERR_MEM_LEAK) (void)PetscErrorMemoryMessage(n);
  else if (n == PETSC_ERR_SUP) {
    (void)(*PetscErrorPrintf)("%s() at %s:%d\n", fun, file, line);
    (void)(*PetscErrorPrintf)("No support for this operation for this object type!\n");
    (void)(*PetscErrorPrintf)("%s\n", mess);
  } else if (n == PETSC_ERR_SIG) (void)(*PetscErrorPrintf)("%s() at %s:%d %s\n", fun, file, line, mess);
  else (void)(*PetscErrorPrintf)("%s() at %s:%d\n    %s\n", fun, file, line, mess);

  PETSCABORT(PETSC_COMM_WORLD, n);
  PetscFunctionReturn(PETSC_SUCCESS);
}
