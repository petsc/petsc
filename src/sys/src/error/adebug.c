/*
*/
#include "petsc.h"
#include <signal.h> 
#include <stdio.h>


static char  *Debugger = "gdb", *Display = 0;
static int   Xterm     = 1;
/*@
     PetscSetDebugger - sets options associated with the debugger.

  Input Parameters:
.   debugger - name of debugger, it should be in your path,
.              usually either "dbx" or "gdb"
.   xterm - 1 to indicate debugger should be started in a new xterm
.           0 to start debugger in initial window (zero makes no 
.           sense if you are using more than one processor.)
.   display - name of display to open xterm to, or null.

  See PetscAttachDebugger() and PetscAttachDebuggerErrorHandler().
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
    if (!Xterm) {
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
}

/*@
    PetscAttachDebuggerErrorHandler - error handler that attaches a
      a debugger to the running process when an error is detected.
      Useful for exaimining variables etc.

  Use:
.  PetscDefaultErrorHandler for tracebacks
,  PetscAbortErrorHandler for when you are already in the debugger.
@*/
int PetscAttachDebuggerErrorHandler(int line,char* file,char* mess,int num)
{
  int ierr = PetscAttachDebugger();
  if (ierr) {
    fprintf(stderr,"Error %s at %d in %s (aborting)\n",mess,line,file);
    exit(num);
  }
  return 0;
}
