#ifndef lint
static char vcid[] = "$Id: adebug.c,v 1.28 1995/10/01 21:51:51 bsmith Exp bsmith $";
#endif

#include "petsc.h"               /*I   "petsc.h"   I*/
#include <signal.h> 
#include <stdio.h>
#include <unistd.h>
#if defined(HAVE_STDLIB_H)
#include <stdlib.h>
#endif
#include "pinclude/petscfix.h"

static char  *Debugger = "gdb", *Display = 0;
static int   Xterm     = 1;
/*@C
   PetscSetDebugger - Sets options associated with the debugger.

   Input Parameters:
.  debugger - name of debugger, which should be in your path,
              usually either "dbx", "gdb" or "xxgdb" or "xdb" on the HP-UX

.   xterm - flag to indicate debugger window, set to one of:
$     1 to indicate debugger should be started in a new xterm
$     0 to start debugger in initial window (zero makes no 
.           sense when using more than one processor.)
.   display - name of display for opening xterm, or null.

.keywords: Set, debugger, options

.seealso: PetscAttachDebugger(), PetscAttachDebuggerErrorHandler()
@*/
int PetscSetDebugger(char *debugger, int xterm,char *display)
{
  if (debugger) Debugger = debugger;
  Xterm    = xterm;
  if (Display) {PETSCFREE(Display); Display = 0;}
  if (display) {
    int len = PetscStrlen(display)+1;
    Display = (char *) PETSCMALLOC(len*sizeof(char)); if (!Display) return 1;
    PetscStrcpy(Display,display);
  }
  return 0;
}

extern char *OptionsGetProgramName();
 

/*@C
   PetscAttachDebugger - Attaches the debugger to the running process.

.keywords: attach, debugger

.seealso: PetscSetDebugger()
@*/
int PetscAttachDebugger()
{
  int   child;
  char *program = OptionsGetProgramName();
#if defined(PARCH_t3d)
  fprintf(stderr,"PETSC ERROR: Cray t3d cannot start debugger\n");
  MPI_Finalize();
  exit(0);
#else
  if (!program) {
    fprintf(stderr,"PETSC ERROR: Cannot determine program name\n");
    return 1;
  }
  child = fork(); 
  if (child <0) {
    fprintf(stderr,"PETSC ERROR: Error attaching debugger\n");
    return -1;
  }
  if (child) { /* I am the parent will run the debugger */
    char  *args[9],pid[9];
#if !defined(PARCH_rs6000)
    kill(child,SIGSTOP);
#endif
    sprintf(pid,"%d",child); 
    if (!PetscStrcmp(Debugger,"xxgdb")) {
      args[1] = program; args[2] = pid; args[3] = "-display";
      args[0] = Debugger; args[4] = Display; args[5] = 0;
      fprintf(stderr,"PETSC: Attaching %s to %s %s\n",args[0],args[1],pid);
      if (execvp(args[0], args)  < 0) {
        perror("Unable to start debugger");
        exit(0);
      }
    }
    else if (!Xterm) {
      args[1] = program; args[2] = pid; args[3] = 0;
      args[0] = Debugger;
#if defined(PARCH_IRIX)
      if (!PetscStrcmp(Debugger,"dbx")) {
        args[1] = "-p";
        args[2] = pid;
        args[3] = program;
        args[4] = 0;
      }
#elif defined(PARCH_hpux)
      if (!PetscStrcmp(Debugger,"xdb")) {
        args[3] = program;
        args[1] = "-P";
        args[2] = pid;
        args[4] = 0;
      }
#elif defined(PARCH_rs6000)
      if (!PetscStrcmp(Debugger,"dbx")) {
        args[1] = "-a";
        args[2] = pid;
        args[3] = 0;
      }
#elif defined(PARCH_alpha)
      if (!PetscStrcmp(Debugger,"dbx")) {
        args[1] = "-pid";
        args[2] = pid;
        args[3] = program;
        args[4] = 0;
      }
#endif
      fprintf(stderr,"PETSC: Attaching %s to %s of pid %s\n",Debugger,
                                                                program,pid);
      if (execvp(args[0], args)  < 0) {
        perror("Unable to start debugger");
        exit(0);
      }
    }
    else {
      if (!Display) {
        args[0] = "xterm";  args[1] = "-e"; 
        args[2] = Debugger; args[3] = program; 
        args[4] = pid;      args[5] = 0;
#if defined(PARCH_IRIX)
        if (!PetscStrcmp(Debugger,"dbx")) {
          args[3] = "-p";
          args[4] = pid;
          args[5] = program;
          args[6] = 0;
        }
#elif defined(PARCH_hpux)
        if (!PetscStrcmp(Debugger,"xdb")) {
          args[5] = program;
          args[3] = "-P";
          args[4] = pid;
          args[6] = 0;
        }
#elif defined(PARCH_rs6000)
        if (!PetscStrcmp(Debugger,"dbx")) {
          args[3] = "-a";
          args[4] = pid;
          args[5] = 0;
        }
#elif defined(PARCH_alpha)
      if (!PetscStrcmp(Debugger,"dbx")) {
        args[3] = "-pid";
        args[4] = pid;
        args[5] = program;
        args[6] = 0;
      }
#endif
        fprintf(stderr,"PETSC: Attaching %s to %s on pid %s\n",Debugger,
                program,pid);
      }
      else {
        args[0] = "xterm";  args[1] = "-d";
        args[2] = Display;  args[3] = "-e";
        args[4] = Debugger; args[5] = program;
        args[6] = pid;      args[7] = 0;
#if defined(PARCH_IRIX)
        if (!PetscStrcmp(Debugger,"dbx")) {
          args[5] = "-p";
          args[6] = pid;
          args[7] = program;
          args[8] = 0;
        }
#elif defined(PARCH_hpux)
        if (!PetscStrcmp(Debugger,"xdb")) {
          args[7] = program;
          args[5] = "-P";
          args[6] = pid;
          args[8] = 0;
        }
#elif defined(PARCH_rs6000)
        if (!PetscStrcmp(Debugger,"dbx")) {
          args[5] = "-a";
          args[6] = pid;
          args[7] = 0;
        }
#elif defined(PARCH_alpha)
      if (!PetscStrcmp(Debugger,"dbx")) {
        args[5] = "-pid";
        args[6] = pid;
        args[7] = program;
        args[8] = 0;
      }
#endif
      fprintf(stderr,"PETSC: Attaching %s to %s of pid %s on display %s\n",
              Debugger,program,pid,Display);
      }

      if (execvp("xterm", args)  < 0) {
        perror("Unable to start debugger");
        exit(0);
      }
    }
  }
  else { /* I am the child, continue with user code */
#if defined(PARCH_hpux) || defined(PARCH_IRIX)
    { 
      double x = 1.0;
      int i=10000000;
        while (i--) x++ ; /* cannot attach to sleeper */
    }
#else
    sleep(10);
#endif
    return 0;
  }
#endif
  return 0;
}

/*@C
   PetscAttachDebuggerErrorHandler - Error handler that attaches a
   a debugger to the running process when an error is detected.
   This routine is useful for examining variables, etc. 

   Input Parameters:
.  line - the line number of the error (indicated by __LINE__)
.  file - the file in which the error was detected (indicated by __FILE__)
.  dir - the directory of the file (indicated by __DIR__)
.  message - an error text string, usually just printed to the screen
.  number - the user-provided error number
.  ctx - error handler context

   Options Database Keys:
$   -on_error_attach_debugger [noxterm,dbx,xxgdb]
$       [-display name]

   Notes:
   By default the GNU debugger, gdb, is used.  Alternatives are dbx and
   xxgdb.

   Most users need not directly employ this routine and the other error 
   handlers, but can instead use the simplified interface SETERR, which has 
   the calling sequence
$     SETERRQ(number,message)

   Notes for experienced users:
   Use PetscPushErrorHandler() to set the desired error handler.  The
   currently available PETSc error handlers are
$    PetscDefaultErrorHandler()
$    PetscAttachDebuggerErrorHandler()
$    PetscAbortErrorHandler()
   or you may write your own.

.keywords: attach, debugger, error, handler

.seealso:  PetscPushErrorHandler(), PetscDefaultErrorHandler(), 
           PetscAbortErrorHandler()
@*/
int PetscAttachDebuggerErrorHandler(int line,char* dir,char* file,int num,char* mess,
                                    void *ctx)
{
  int ierr;
  fprintf(stderr,"Error %s at %d in %s (aborting)\n",mess,line,file);
  ierr = PetscAttachDebugger();
  if (ierr) { /* hopeless so get out */
    exit(num);
  }
  return 0;
}
