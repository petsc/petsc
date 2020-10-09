
#include <petscsys.h>           /*I "petscsys.h" I*/

/*@C
   PetscMPIAbortErrorHandler - Calls PETSCABORT and exits.

   Not Collective

   Input Parameters:
+  comm - communicator over which error occurred
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - PETSC_ERROR_INITIAL if error just detected, otherwise PETSC_ERROR_REPEAT
-  ctx - error handler context

   Level: developer

   Notes:
   Most users need not directly employ this routine and the other error
   handlers, but can instead use the simplified interface SETERRQ, which has
   the calling sequence
$     SETERRQ(comm,n,p,mess)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers include PetscTraceBackErrorHandler(),
   PetscMPIAbortErrorHandler(), PetscAttachDebuggerErrorHandler(), and PetscAbortErrorHandler().

.seealso:  PetscError(), PetscPushErrorHandler(), PetscPopErrorHandler(), PetscAttachDebuggerErrorHandler(),
           PetscAbortErrorHandler(), PetscMPIAbortErrorHandler(), PetscTraceBackErrorHandler(), PetscEmacsClientErrorHandler(), PetscReturnErrorHandler()
 @*/
PetscErrorCode  PetscMPIAbortErrorHandler(MPI_Comm comm,int line,const char *fun,const char *file,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  PetscBool      flg1 = PETSC_FALSE,flg2 = PETSC_FALSE,flg3 = PETSC_FALSE;
  PetscLogDouble mem,rss;

  PetscFunctionBegin;
  if (!mess) mess = " ";

  if (n == PETSC_ERR_MEM) {
    (*PetscErrorPrintf)("%s() line %d in %s\n",fun,line,file);
    (*PetscErrorPrintf)("Out of memory. This could be due to allocating\n");
    (*PetscErrorPrintf)("too large an object or bleeding by not properly\n");
    (*PetscErrorPrintf)("destroying unneeded objects.\n");
    PetscMallocGetCurrentUsage(&mem); PetscMemoryGetCurrentUsage(&rss);
    PetscOptionsGetBool(NULL,NULL,"-malloc_dump",&flg1,NULL);
    PetscOptionsGetBool(NULL,NULL,"-malloc_view",&flg2,NULL);
    PetscOptionsHasName(NULL,NULL,"-malloc_view_threshold",&flg3);
    if (flg2 || flg3) PetscMallocView(stdout);
    else {
      (*PetscErrorPrintf)("Memory allocated %.0f Memory used by process %.0f\n",mem,rss);
      if (flg1) PetscMallocDump(stdout);
      else (*PetscErrorPrintf)("Try running with -malloc_dump or -malloc_view for info.\n");
    }
  } else if (n == PETSC_ERR_SUP) {
    (*PetscErrorPrintf)("%s() line %d in %s\n",fun,line,file);
    (*PetscErrorPrintf)("No support for this operation for this object type!\n");
    (*PetscErrorPrintf)("%s\n",mess);
  } else if (n == PETSC_ERR_SIG) (*PetscErrorPrintf)("%s() line %d in %s %s\n",fun,line,file,mess);
  else (*PetscErrorPrintf)("%s() line %d in %s\n    %s\n",fun,line,file,mess);

  PETSCABORT(PETSC_COMM_WORLD,n);
  PetscFunctionReturn(0);
}

