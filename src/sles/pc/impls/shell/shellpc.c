#ifndef lint
static char vcid[] = "$Id: shellpc.c,v 1.21 1996/01/26 01:42:02 curfman Exp bsmith $";
#endif

/*
   This provides a simple shell for Fortran (and C programmers) to 
  create their own preconditioner without writing much interface code.
*/

#include "pcimpl.h"        /*I "pc.h" I*/
#include "vec/vecimpl.h"  

typedef struct {
  void *ctx, *ctxrich;    /* user provided contexts for preconditioner */
  int  (*apply)(void *,Vec,Vec);
  int  (*applyrich)(void *,Vec,Vec,Vec,int);
  char *name;
} PC_Shell;

static int PCApply_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell *shell;
  int      ierr;

  shell = (PC_Shell *) pc->data;
  ierr = (*shell->apply)(shell->ctx,x,y); CHKERRQ(ierr);
  return 0;
}
static int PCApplyRichardson_Shell(PC pc,Vec x,Vec y,Vec w,int it)
{
  PC_Shell *shell;
  shell = (PC_Shell *) pc->data;
  return (*shell->applyrich)(shell->ctx,x,y,w,it);
}
static int PCDestroy_Shell(PetscObject obj)
{
  PC       pc = (PC) obj;
  PC_Shell *shell = (PC_Shell *) pc->data;
  PetscFree(shell);
  return 0;
}
static int PCView_Shell(PetscObject obj,Viewer viewer)
{
  PC         pc = (PC)obj;
  PC_Shell   *jac = (PC_Shell *) pc->data;
  FILE       *fd;
  int        ierr;

  ierr = ViewerFileGetPointer(viewer,&fd); CHKERRQ(ierr);
  if (jac->name) MPIU_fprintf(pc->comm,fd,"    Shell: %s\n", jac->name);
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
.             PCSetType(pc,PC_Shell);
.             PC_ShellSetApply(pc,mult,ctx);

*/
int PCCreate_Shell(PC pc)
{
  PC_Shell *shell;

  pc->destroy    = PCDestroy_Shell;
  shell          = PetscNew(PC_Shell); CHKPTRQ(shell);
  pc->data       = (void *) shell;
  pc->apply      = PCApply_Shell;
  pc->applyrich  = 0;
  pc->setup      = 0;
  pc->type       = PCSHELL;
  pc->view       = PCView_Shell;
  pc->name       = 0;
  shell->apply   = 0;
  return 0;
}

/*@C
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
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  shell        = (PC_Shell *) pc->data;
  shell->apply = apply;
  shell->ctx   = ptr;
  return 0;
}

/*@C
   PCShellSetName - Sets an optional name to associate with a shell
   preconditioner.

   Input Parameters:
.  pc - the preconditioner context
.  name - character string describing shell preconditioner

.keywords: PC, shell, set, name, user-provided

.seealso: PCShellGetName()
@*/
int PCShellSetName(PC pc,char *name)
{
  PC_Shell *shell;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  shell       = (PC_Shell *) pc->data;
  shell->name = name;
  return 0;
}

/*@C
   PCShellGetName - Gets an optional name that the user has set for a shell
   preconditioner.

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  name - character string describing shell preconditioner

.keywords: PC, shell, get, name, user-provided

.seealso: PCShellSetName()
@*/
int PCShellGetName(PC pc,char **name)
{
  PC_Shell *shell;
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  shell = (PC_Shell *) pc->data;
  *name  = shell->name;
  return 0;
}

/*@C
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
  PETSCVALIDHEADERSPECIFIC(pc,PC_COOKIE);
  shell            = (PC_Shell *) pc->data;
  pc->applyrich    = PCApplyRichardson_Shell;
  shell->applyrich = apply;
  shell->ctxrich   = ptr;
  return 0;
}
