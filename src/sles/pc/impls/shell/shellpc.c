
/*
   This provides a simple shell for Fortran (and C programmers) to 
  create a very simple preconditioner class for use with KSP without coding 
  mush of anything.
*/

#include "petsc.h"
#include "pcimpl.h"        /*I "pc.h" I*/
#include "vec/vecimpl.h"  

typedef struct {
  void *ctx;
  int  (*apply)(void *,Vec,Vec);
} PCShell;

static int PCShellApply(PC pc,Vec x,Vec y)
{
  PCShell *shell;
  shell = (PCShell *) pc->data;
  return (*shell->apply)(shell->ctx,x,y);
}
static int PCShellDestroy(PetscObject obj)
{
  PC      pc = (PC) obj;
  PCShell *shell;
  shell = (PCShell *) pc->data;
  FREE(shell); FREE(pc);
  return 0;
}
  

/*
   PCShellCreate - creates a new preconditioner class for use with your 
          own private data storage format. This is intended to 
          provide a simple class to use with KSP. You should 
          not use this if you plan to make a complete class.


  Usage:
.             int (*mult)(void *,Vec,Vec);
.             PCCreate(&pc);
.             PCSetMethod(pc,PCSHELL);
.             PCShellSetApply(pc,mult,ctx);

*/
int PCiShellCreate(PC pc)
{
  PCShell *shell;

  pc->destroy    = PCShellDestroy;
  shell          = NEW(PCShell); CHKPTR(shell);
  pc->data       = (void *) shell;
  pc->apply      = PCShellApply;
  pc->setup      = 0;
  pc->type       = PCSHELL;
  shell->apply   = 0;
  return 0;
}

/*@
   PCShellSetApply - sets routine to use as preconditioner.

  Input Parameters:
.  pc - the preconditioner context
.  mult - the application routine.
.  ctx - pointer to data needed by application routine

  Keywords: preconditioner, user-provided
@*/
int PCShellSetApply(PC pc, int (*mult)(void*,Vec,Vec),void *ctx)
{
  PCShell *shell;
  VALIDHEADER(pc,PC_COOKIE);
  shell        = (PCShell *) pc->data;
  shell->apply = mult;
  shell->ctx   = ctx;
  return 0;
}
