
/*
       The default error handlers and code that allows one to change
   error handlers.
*/
#include <petscsys.h>           /*I "petscsys.h" I*/
#if defined(PETSC_HAVE_STDLIB_H)
#include <stdlib.h>
#endif

#undef __FUNCT__  
#define __FUNCT__ "PetscAbortErrorHandler" 
/*@C
   PetscAbortErrorHandler - Error handler that calls abort on error. 
   This routine is very useful when running in the debugger, because the 
   user can look directly at the stack frames and the variables.

   Not Collective

   Input Parameters:
+  comm - communicator over which error occurred
.  line - the line number of the error (indicated by __LINE__)
.  func - function where error occured (indicated by __FUNCT__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __SDIR__)
.  mess - an error text string, usually just printed to the screen
.  n - the generic error number
.  p - specific error number
-  ctx - error handler context

   Options Database Keys:
+  -on_error_abort - Activates aborting when an error is encountered
-  -start_in_debugger [noxterm,dbx,xxgdb]  [-display name] - Starts all
    processes in the debugger and uses PetscAbortErrorHandler().  By default the 
    debugger is gdb; alternatives are dbx and xxgdb.

   Level: developer

   Notes:
   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERRQ, which
   has the calling sequence
$     SETERRQ(comm,number,mess)
   or its variants, SETERRQ1(number,formatstring,arg1), SETERRQ2(), ... that
   allow including arguments in the message.

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers include PetscTraceBackErrorHandler(),
   PetscAttachDebuggerErrorHandler(), and PetscAbortErrorHandler().

   Concepts: error handler^aborting
   Concepts: aborting on error

.seealso: PetscPushErrorHandler(), PetscTraceBackErrorHandler(), 
          PetscAttachDebuggerErrorHandler()
@*/
PetscErrorCode  PetscAbortErrorHandler(MPI_Comm comm,int line,const char *fun,const char *file,const char* dir,PetscErrorCode n,PetscErrorType p,const char *mess,void *ctx)
{
  PetscFunctionBegin;
  (*PetscErrorPrintf)("%s() line %d in %s%s %s\n",fun,line,dir,file,mess);
  abort(); 
  PetscFunctionReturn(0);
}

