#ifndef lint
static char vcid[] = "$Id: shell.c,v 1.10 1995/04/16 03:43:14 curfman Exp curfman $";
#endif

/*
   This provides a simple shell for Fortran (and C programmers) to 
  create a very simple preconditioner class for use with KSP without coding 
  mush of anything.
*/

#include "petsc.h"
#include "pcimpl.h"        /*I "pc.h" I*/
#include "vec/vecimpl.h"  

typedef struct {
  void *ctx,*ctxrich;
  int  (*apply)(void *,Vec,Vec);
  int  (*applyrich)(void *,Vec,Vec,Vec,int);
} PC_Shell;

static int PCApply_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell *shell;
  shell = (PC_Shell *) pc->data;
  return (*shell->apply)(shell->ctx,x,y);
}
static int PCApplyRichardson_Shell(PC pc,Vec x,Vec y,Vec w,int it)
{
  PC_Shell *shell;
  shell = (PC_Shell *) pc->data;
  return (*shell->applyrich)(shell->ctx,x,y,w,it);
}
static int PCDestroy_Shell(PetscObject obj)
{
  PC      pc = (PC) obj;
  PC_Shell *shell;
  shell = (PC_Shell *) pc->data;
  FREE(shell);
  PLogObjectDestroy(pc);
  PETSCHEADERDESTROY(pc);
  return 0;
}

/*
   PCCreate_Shell - creates a new preconditioner class for use with your 
          own private data storage format. This is intended to 
          provide a simple class to use with KSP. You should 
          not use this if you plan to make a complete class.


  Usage:
.             int (*mult)(void *,Vec,Vec);
.             PCCreate(comm,&pc);
.             PCSetMethod(pc,PC_Shell);
.             PC_ShellSetApply(pc,mult,ctx);

*/
int PCCreate_Shell(PC pc)
{
  PC_Shell *shell;

  pc->destroy    = PCDestroy_Shell;
  shell          = NEW(PC_Shell); CHKPTR(shell);
  pc->data       = (void *) shell;
  pc->apply      = PCApply_Shell;
  pc->applyrich  = 0;
  pc->setup      = 0;
  pc->type       = PCSHELL;
  shell->apply   = 0;
  return 0;
}

/*@
   PCShellSetApply - Sets routine to use as preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  apply - the application-provided preconditioning routine
.  ptr - pointer to data needed by this routine

   Calling sequence of apply:
   int apply (void *ptr,Vec xin,Vec xout)
.  ptr - the application context
.  xin - input vector
.  xout - output vector

.keywords: PC, shell, set, apply, user-provided

.seealso: PCShellSetApplyRichardson()
@*/
int PCShellSetApply(PC pc, int (*apply)(void*,Vec,Vec),void *ptr)
{
  PC_Shell *shell;
  VALIDHEADER(pc,PC_COOKIE);
  shell        = (PC_Shell *) pc->data;
  shell->apply = apply;
  shell->ctx   = ptr;
  return 0;
}

/*@
   PCShellSetApplyRichardson - Sets routine to use as preconditioner
   in Richardson iteration.

   Input Parameters:
.  pc - the preconditioner context
.  apply - the application-provided preconditioning routine
.  ptr - pointer to data needed by this routine

   Calling sequence of apply:
   int apply (void *ptr,Vec x,Vec b,Vec r,int maxits)
.  ptr - the application context
.  x - current iterate
.  b - right-hand-side
.  r - residual
.  maxits - maximum number of iterations

.keywords: PC, shell, set, apply, Richardson, user-provided

.seealso: PCShellSetApply()
@*/
int PCShellSetApplyRichardson(PC pc, int (*apply)(void*,Vec,Vec,Vec,int),
                              void *ptr)
{
  PC_Shell *shell;
  VALIDHEADER(pc,PC_COOKIE);
  shell            = (PC_Shell *) pc->data;
  pc->applyrich    = PCApplyRichardson_Shell;
  shell->applyrich = apply;
  shell->ctxrich   = ptr;
  return 0;
}
