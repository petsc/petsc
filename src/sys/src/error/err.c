
#include "petsc.h"
#include <stdio.h>  /*I <stdio.h> I*/

/*@
    PetscAbortErrorHandler - an error handler routine that calls 
        abort on error. This is very useful when running in the 
        debugger, because you can look directly at the stack frames
        and the variables.

  Use:
.  PetscDefaultErrorHandler() for generating tracebacks
.  PetscAttachDebuggerErrorHandler() for automatically attaching the 
.          debugger when an error is detected.
@*/
int PetscAbortErrorHandler(int line,char *file,char *message,int number)
{
  abort(0);
}
/*@
    PetscDefaultErrorHandler - an error handler routine that generates
        a traceback on error detection.

  Use:
.  PetscAbortErrorHandler() for when you are running in the debugger and
.         would like it to stop at the error, so you may examine variables.
.  PetscAttachDebuggerErrorHandler() for automatically attaching the 
.          debugger when an error is detected.
@*/
int PetscDefaultErrorHandler(int line,char *file,char *message,int number)
{
  fprintf(stderr,"%s %d %s %d\n",file,line,message,number);
  return number;
}

static int (*errorhandler)(int,char*,char*,int) = PetscDefaultErrorHandler;

/*@
    PetscSetErrorHandler - Sets a function to be called on errors.

  Input Parameters:
.  func - error handler

  Call sequence of function:
.  int func(int linenumber,char *filename,char* errormessage,int errorno);
@*/
int PetscSetErrorHandler(int (*handler)(int,char*,char*,int) )
{
  errorhandler = handler;
  return 0;
}
/*@
    PetscErrorHandler - Handles error. Will eventually call a (possibly)
        user provided function.

  Input Parameters:
.  line,file - the linenumber and file the error was detected in
.  message - a text string usually just printed to the screen
.  number - the user provided error number.
@*/
int PetscErrorHandler(int line,char *file,char *message,int number)
{
  return (*errorhandler)(line,file,message,number);
}
