
#include <petscsys.h> /*I "petscsys.h" I*/

/*@C
   PetscMPIAbortErrorHandler - Calls PETSCABORT and exits.

   Not Collective

   Input Parameters:
+  comm - communicator over which error occurred
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - `PETSC_ERROR_INITIAL` if error just detected, otherwise `PETSC_ERROR_REPEAT`
-  ctx - error handler context

   Level: developer

   Note:
   Users do not directly call this routine

   Use `PetscPushErrorHandler()` to set the desired error handler.  The
   currently available PETSc error handlers include `PetscTraceBackErrorHandler()`,
   `PetscMPIAbortErrorHandler()`, `PetscAttachDebuggerErrorHandler()`, and `PetscAbortErrorHandler()`.

.seealso: `PetscError()`, `PetscPushErrorHandler()`, `PetscPopErrorHandler()`, `PetscAttachDebuggerErrorHandler()`,
          `PetscAbortErrorHandler()`, `PetscMPIAbortErrorHandler()`, `PetscTraceBackErrorHandler()`, `PetscEmacsClientErrorHandler()`, `PetscReturnErrorHandler()`
 @*/
PetscErrorCode PetscMPIAbortErrorHandler(MPI_Comm comm, int line, const char *fun, const char *file, PetscErrorCode n, PetscErrorType p, const char *mess, void *ctx)
{
  PetscErrorCode ierr;
  PetscBool      flg1 = PETSC_FALSE, flg2 = PETSC_FALSE, flg3 = PETSC_FALSE;
  PetscLogDouble mem, rss;

  PetscFunctionBegin;
  if (!mess) mess = " ";

  if (n == PETSC_ERR_MEM) {
    ierr = (*PetscErrorPrintf)("%s() at %s:%d\n", fun, file, line);
    ierr = (*PetscErrorPrintf)("Out of memory. This could be due to allocating\n");
    ierr = (*PetscErrorPrintf)("too large an object or bleeding by not properly\n");
    ierr = (*PetscErrorPrintf)("destroying unneeded objects.\n");
    ierr = PetscMallocGetCurrentUsage(&mem);
    ierr = PetscMemoryGetCurrentUsage(&rss);
    ierr = PetscOptionsGetBool(NULL, NULL, "-malloc_dump", &flg1, NULL);
    ierr = PetscOptionsGetBool(NULL, NULL, "-malloc_view", &flg2, NULL);
    ierr = PetscOptionsHasName(NULL, NULL, "-malloc_view_threshold", &flg3);
    if (flg2 || flg3) ierr = PetscMallocView(stdout);
    else {
      ierr = (*PetscErrorPrintf)("Memory allocated %.0f Memory used by process %.0f\n", mem, rss);
      if (flg1) ierr = PetscMallocDump(stdout);
      else ierr = (*PetscErrorPrintf)("Try running with -malloc_dump or -malloc_view for info.\n");
    }
  } else if (n == PETSC_ERR_SUP) {
    ierr = (*PetscErrorPrintf)("%s() at %s:%d\n", fun, file, line);
    ierr = (*PetscErrorPrintf)("No support for this operation for this object type!\n");
    ierr = (*PetscErrorPrintf)("%s\n", mess);
  } else if (n == PETSC_ERR_SIG) ierr = (*PetscErrorPrintf)("%s() at %s:%d %s\n", fun, file, line, mess);
  else ierr = (*PetscErrorPrintf)("%s() at %s:%d\n    %s\n", fun, file, line, mess);

  (void)ierr;
  PETSCABORT(PETSC_COMM_WORLD, n);
  PetscFunctionReturn(PETSC_SUCCESS);
}
