#ifndef lint
static char vcid[] = "$Id: adebug.c,v 1.8 1995/03/06 04:31:52 bsmith Exp curfman $";
#endif/*
*/
#include "petsc.h"
#include <signal.h> 
#include <stdio.h>
#include <unistd.h>

static char  *Debugger = "gdb", *Display = 0;
static int   Xterm     = 1;
/*@
   PetscSetDebugger - Sets options associated with the debugger.

   Input Parameters:
.  debugger - name of debugger, which should be in your path,
              usually either "dbx", "gdb" or "xxgdb"
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
  if (Display) {FREE(Display); Display = 0;}
  if (display) {
    int len = strlen(display)+1;
    Display = (char *) MALLOC(len*sizeof(char)); if (!Display) return 1;
    strcpy(Display,display);
  }
  return 0;
}

extern char *OptionsGetProgramName();
 
#if defined(PARCH_sun4) && defined(__cplusplus)
extern "C" {
int exit(int);
};
#endif
/*@
    PetscAttachDebugger - Attaches debugger to the running process.

    See: PetscSetDebugger().
@*/
int PetscAttachDebugger()
{
  int   child;
  char *program = OptionsGetProgramName();
  if (!program) {
    fprintf(stderr,"Cannot determine program name\n");
    fprintf(stderr,"Program probably does not have a OptionsCreate()\n");
    return 1;
  }
  child = fork(); 
  if (child <0) {
    fprintf(stderr,"Error attaching debugger\n");
    return -1;
  }
  if (child) { /* I am the parent will run the debugger */
    char  *args[8],pid[8];
    kill(child,SIGSTOP);
    sprintf(pid,"%d",child); 
    if (!strcmp(Debugger,"xxgdb")) {
      args[1] = program; args[2] = pid; args[3] = "-display";
      args[0] = Debugger; args[4] = Display; args[5] = 0;
      fprintf(stderr,"Attaching %s to %s %s\n",args[0],args[1],pid);
      if (execvp(args[0], args)  < 0) {
        perror("Unable to start debugger");
        exit(0);
      }
    }
    else if (!Xterm) {
      args[1] = program; args[2] = pid; args[3] = 0;
      args[0] = Debugger;
      fprintf(stderr,"Attaching %s to %s %s\n",args[0],args[1],pid);
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
        fprintf(stderr,"Attaching %s to %s %s\n",args[2],args[3],pid);
      }
      else {
        args[0] = "xterm";  args[1] = "-d";
        args[2] = Display;  args[3] = "-e";
        args[4] = Debugger; args[5] = program;
        args[6] = pid;      args[7] = 0;
        fprintf(stderr,"Attaching %s to %s %s on %s\n",args[4],args[5],pid,
                       Display);  
      }

      if (execvp("xterm", args)  < 0) {
        perror("Unable to start debugger");
        exit(0);
      }
    }
  }
  else { /* I am the child, continue with user code */
    sleep(10);
    return 0;
  }
  return 0;
}

/*@
    PetscAttachDebuggerErrorHandler - Error handler that attaches a
      a debugger to the running process when an error is detected.
      Useful for exaimining variables etc.

  Use:
.  PetscDefaultErrorHandler for tracebacks
,  PetscAbortErrorHandler for when you are already in the debugger.
@*/
int PetscAttachDebuggerErrorHandler(int line,char* dir,char* file,char* mess,
                                    int num,void *ctx)
{
  int ierr = PetscAttachDebugger();
  if (ierr) {
    fprintf(stderr,"Error %s at %d in %s (aborting)\n",mess,line,file);
    exit(num);
  }
  return 0;
}
