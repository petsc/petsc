/*
*/
#include "include/petscimp.h"
#include <signal.h> 
#include <stdio.h>


int PetscDebuggerError(p,err,s,l,f)
PetscStruct p;
int         err,l;
char        *s,*f;
{
  char *program = (char *) p->errctx;

  PetscAttachDebugger(program);
  return 0;
}
int PetscAttachDebugger(program)
char *program;
{
  int   child;
  
  child = fork(); 
  if (child) { /* I am the parent will run the debugger */
    char  *args[4],pid[8];
    kill(child,SIGSTOP);
    sprintf(pid,"%d",child); 
    args[0] = "dbx"; args[1] = program; args[2] = pid; args[3] = 0;
    fprintf(stderr,"Attaching dbx to %s %s\n",args[1],pid);
    if (execvp("/usr/ucb/dbx", args)  < 0) {
      perror("Unable to start debugger");
      exit(0);
    }
  }
  else { /* I am the child, continue with user code */
    sleep(10);
  }
}
