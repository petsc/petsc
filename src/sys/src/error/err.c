
#include "petsc.h"
#include <stdio.h>  /*I <stdio.h> I*/

struct EH {
  int    cookie;
  int    (*handler)(int, char*,char *,char *,int,void *);
  void   *ctx;
  struct EH* previous;
};

static struct EH* eh = 0;


#if defined(PARCH_sun4) && defined(__cplusplus)
extern "C" {
int abort();
};
#endif
/*@
    PetscAbortErrorHandler - Error handler routine that calls 
        abort on error. This is very useful when running in the 
        debugger, because you can look directly at the stack frames
        and the variables.

  Use:
.  PetscDefaultErrorHandler() for generating tracebacks
.  PetscAttachDebuggerErrorHandler() for automatically attaching the 
.          debugger when an error is detected.
@*/
int PetscAbortErrorHandler(int line,char* dir,char *file,char *message,
                           int number,void *ctx)
{
  abort(); return 0;
}
/*@
    PetscDefaultErrorHandler - Error handler routine that generates
        a traceback on error detection.

  Use:
.  PetscAbortErrorHandler() for when you are running in the debugger and
.         would like it to stop at the error, so you may examine variables.
.  PetscAttachDebuggerErrorHandler() for automatically attaching the 
.          debugger when an error is detected.
@*/
int PetscDefaultErrorHandler(int line,char *dir,char *file,char *message,
                             int number,void *ctx)
{
  if (!dir) fprintf(stderr,"%s %d %s %d\n",file,line,message,number);
  else      fprintf(stderr,"%s%s %d %s %d\n",dir,file,line,message,number);
  return number;
}

/*@
    PetscPushErrorHandler - Sets a function to be called on errors.

  Input Parameters:
.  func - error handler

  Call sequence of function:
.  int func(int line,char *dir,char *filename,char* errormessage,int errorno);
@*/
int PetscPushErrorHandler(int (*handler)(int,char*,char*,char*,int,void*),
                          void *ctx )
{
  struct  EH *neweh = NEW(struct EH); CHKPTR(neweh);
  if (eh) {neweh->previous = eh;} 
  else {neweh->previous = 0;}
  neweh->handler = handler;
  neweh->ctx     = ctx;
  eh = neweh;
  return 0;
}
int PetscPopErrorHandler()
{
  struct EH *tmp;
  if (!eh) return 0;
  tmp = eh;
  eh = eh->previous;
  FREE(tmp);

  return 0;
}
/*@
    PetscError - Called when error is detected, usually called through
                 the macro SETERR().

  Input Parameters:
.  line,file - the linenumber and file the error was detected in
.  dir - the directory as indicated by __DIR__
.  message - a text string usually just printed to the screen
.  number - the user provided error number.
@*/
int PetscError(int line,char *dir,char *file,char *message,int number)
{
  if (!eh) return PetscDefaultErrorHandler(line,dir,file,message,number,0);
  else  return (*eh->handler)(line,dir,file,message,number,eh->ctx);
}

/*
     Useful functions for debugging
*/
int IntView(int N,int* idx,Viewer viewer)
{
  int j,i,n = N/20, p = N % 20;

  for ( i=0; i<n; i++ ) {
    printf("%d:",20*i);
    for ( j=0; j<20; j++ ) {
       printf(" %d",idx[i*20+j]);
    }
    printf("\n");
  }
  if (p) {
    printf("%d:",20*n);
    for ( i=0; i<p; i++ ) { printf(" %d",idx[20*n+i]);}
    printf("\n");
  }
  return 0;
}
int DoubleView(int N,double* idx,Viewer viewer)
{
  int j,i,n = N/5, p = N % 5;

  for ( i=0; i<n; i++ ) {
    printf("%d:",5*i);
    for ( j=0; j<5; j++ ) {
       printf(" %6.4e",idx[i*5+j]);
    }
    printf("\n");
  }
  if (p) {
    printf("%d:",5*n);
    for ( i=0; i<p; i++ ) { printf(" %6.4e",idx[5*n+i]);}
    printf("\n");
  }
  return 0;
}

