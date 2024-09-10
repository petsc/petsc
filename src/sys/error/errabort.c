/*
       The default error handlers and code that allows one to change
   error handlers.
*/
#include <petscsys.h> /*I "petscsys.h" I*/

/*@C
  PetscAbortErrorHandler - Error handler that calls abort on error.
  This routine is very useful when running in the debugger, because the
  user can look directly at the stack frames and the variables where the error occurred

  Not Collective, No Fortran Support

  Input Parameters:
+ comm - communicator over which error occurred
. line - the line number of the error (usually indicated by `__LINE__` in the calling routine)
. fun  - the function name of the calling routine
. file - the file in which the error was detected (usually indicated by `__FILE__` in the calling routine)
. mess - an error text string, usually this is just printed to the screen
. n    - the generic error number
. p    - `PETSC_ERROR_INITIAL` indicates this is the first time the error handler is being called while `PETSC_ERROR_REPEAT` indicates it was previously called
- ctx  - error handler context

  Options Database Keys:
+ -on_error_abort                                          - Activates aborting when an error is encountered
- -start_in_debugger [noxterm,lldb or gdb] [-display name] - Starts all processes in the debugger and uses `PetscAbortErrorHandler()`. By default on Linux the
                                                             debugger is gdb and on macOS it is lldb

  Level: developer

  Notes:
  Users do not directly employ this routine

  Use `PetscPushErrorHandler()` to set the desired error handler.  The
  currently available PETSc error handlers include `PetscTraceBackErrorHandler()`,
  `PetscAttachDebuggerErrorHandler()`, and `PetscAbortErrorHandler()`.

.seealso: `PetscError()`, `PetscPushErrorHandler()`, `PetscPopErrorHander()`, `PetscTraceBackErrorHandler()`,
          `PetscAttachDebuggerErrorHandler()`, `PetscMPIAbortErrorHandler()`, `PetscReturnErrorHandler()`, `PetscEmacsClientErrorHandler()`,
          `PetscErrorType`, `PETSC_ERROR_INITIAL`, `PETSC_ERROR_REPEAT`, `PetscErrorCode`
@*/
PetscErrorCode PetscAbortErrorHandler(MPI_Comm comm, int line, const char *fun, const char *file, PetscErrorCode n, PetscErrorType p, const char *mess, void *ctx)
{
  size_t len;

  PetscFunctionBegin;
  (void)comm;
  (void)p;
  (void)ctx;
  (void)n;
  (void)PetscStrlen(fun, &len);
  if (len) {
    (void)(*PetscErrorPrintf)("PetscAbortErrorHandler: %s() at %s:%d %s\n  To prevent termination, change the error handler using PetscPushErrorHandler()\n", fun, file, line, mess);
  } else {
    (void)(*PetscErrorPrintf)("PetscAbortErrorHandler: %s\n  To prevent termination, change the error handler using PetscPushErrorHandler()\n", mess);
  }
  abort();
  PetscFunctionReturn(PETSC_SUCCESS);
}
