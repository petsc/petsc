#ifdef PETSC_RCS_HEADER
static char vcid[] = "$Id: shellpc.c,v 1.42 1998/04/24 21:21:24 curfman Exp bsmith $";
#endif

/*
   This provides a simple shell for Fortran (and C programmers) to 
  create their own preconditioner without writing much interface code.
*/

#include "src/pc/pcimpl.h"        /*I "pc.h" I*/
#include "src/vec/vecimpl.h"  

typedef struct {
  void *ctx, *ctxrich;    /* user provided contexts for preconditioner */
  int  (*apply)(void *,Vec,Vec);
  int  (*applyrich)(void *,Vec,Vec,Vec,int);
  char *name;
} PC_Shell;

#undef __FUNC__  
#define __FUNC__ "PCApply_Shell"
static int PCApply_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell *shell;
  int      ierr;

  PetscFunctionBegin;
  shell = (PC_Shell *) pc->data;
  ierr = (*shell->apply)(shell->ctx,x,y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApplyRichardson_Shell"
static int PCApplyRichardson_Shell(PC pc,Vec x,Vec y,Vec w,int it)
{
  int      ierr;
  PC_Shell *shell;

  PetscFunctionBegin;
  shell = (PC_Shell *) pc->data;
  ierr = (*shell->applyrich)(shell->ctx,x,y,w,it);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCDestroy_Shell"
static int PCDestroy_Shell(PC pc)
{
  PC_Shell *shell = (PC_Shell *) pc->data;

  PetscFunctionBegin;
  PetscFree(shell);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_Shell"
static int PCView_Shell(PC pc,Viewer viewer)
{
  PC_Shell   *jac = (PC_Shell *) pc->data;
  FILE       *fd;
  int        ierr;
  ViewerType vtype;

  PetscFunctionBegin;
  ierr = ViewerGetType(viewer,&vtype); CHKERRQ(ierr);
  if (vtype  == ASCII_FILE_VIEWER || vtype == ASCII_FILES_VIEWER) {  
    ierr = ViewerASCIIGetPointer(viewer,&fd); CHKERRQ(ierr);
    if (jac->name) PetscFPrintf(pc->comm,fd,"    Shell: %s\n", jac->name);
  } else {
    SETERRQ(1,1,"Viewer type not supported for this object");
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
#undef __FUNC__  
#define __FUNC__ "PCShellSetApply_Shell"
int PCShellSetApply_Shell(PC pc, int (*apply)(void*,Vec,Vec),void *ptr)
{
  PC_Shell *shell;

  PetscFunctionBegin;
  shell        = (PC_Shell *) pc->data;
  shell->apply = apply;
  shell->ctx   = ptr;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCShellSetName_Shell"
int PCShellSetName_Shell(PC pc,char *name)
{
  PC_Shell *shell;

  PetscFunctionBegin;
  shell       = (PC_Shell *) pc->data;
  shell->name = name;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCShellGetName_Shell"
int PCShellGetName_Shell(PC pc,char **name)
{
  PC_Shell *shell;

  PetscFunctionBegin;
  shell  = (PC_Shell *) pc->data;
  *name  = shell->name;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCShellSetApplyRichardson_Shell"
int PCShellSetApplyRichardson_Shell(PC pc, int (*apply)(void*,Vec,Vec,Vec,int),void *ptr)
{
  PC_Shell *shell;

  PetscFunctionBegin;
  shell            = (PC_Shell *) pc->data;
  pc->applyrich    = PCApplyRichardson_Shell;
  shell->applyrich = apply;
  shell->ctxrich   = ptr;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------*/

#undef __FUNC__  
#define __FUNC__ "PCShellSetApply"
/*@C
   PCShellSetApply - Sets routine to use as preconditioner.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  apply - the application-provided preconditioning routine
-  ptr - pointer to data needed by this routine

   Calling sequence of apply:
.vb
   int apply (void *ptr,Vec xin,Vec xout)
.ve

.  ptr - the application context
.  xin - input vector
.  xout - output vector

.keywords: PC, shell, set, apply, user-provided

.seealso: PCShellSetApplyRichardson()
@*/
int PCShellSetApply(PC pc, int (*apply)(void*,Vec,Vec),void *ptr)
{
  int ierr, (*f)(PC,int (*)(void*,Vec,Vec),void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellSetApply_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,apply,ptr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCShellSetName"
/*@C
   PCShellSetName - Sets an optional name to associate with a shell
   preconditioner.

   Not Collective

   Input Parameters:
+  pc - the preconditioner context
-  name - character string describing shell preconditioner

.keywords: PC, shell, set, name, user-provided

.seealso: PCShellGetName()
@*/
int PCShellSetName(PC pc,char *name)
{
  int ierr, (*f)(PC,char *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellSetName_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCShellGetName"
/*@C
   PCShellGetName - Gets an optional name that the user has set for a shell
   preconditioner.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  name - character string describing shell preconditioner

.keywords: PC, shell, get, name, user-provided

.seealso: PCShellSetName()
@*/
int PCShellGetName(PC pc,char **name)
{
  int ierr, (*f)(PC,char **);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellGetName_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,name);CHKERRQ(ierr);
  } else {
    SETERRQ(1,1,"Not shell preconditioner, cannot get name");
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCShellSetApplyRichardson"
/*@C
   PCShellSetApplyRichardson - Sets routine to use as preconditioner
   in Richardson iteration.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  apply - the application-provided preconditioning routine
-  ptr - pointer to data needed by this routine

   Calling sequence of apply:
.vb
   int apply (void *ptr,Vec x,Vec b,Vec r,int maxits)
.ve

+  ptr - the application context
.  x - current iterate
.  b - right-hand-side
.  r - residual
-  maxits - maximum number of iterations

.keywords: PC, shell, set, apply, Richardson, user-provided

.seealso: PCShellSetApply()
@*/
int PCShellSetApplyRichardson(PC pc, int (*apply)(void*,Vec,Vec,Vec,int),void *ptr)
{
  int ierr, (*f)(PC,int (*)(void*,Vec,Vec,Vec,int),void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellSetApplyRichardson_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,apply,ptr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
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
#undef __FUNC__  
#define __FUNC__ "PCCreate_Shell"
int PCCreate_Shell(PC pc)
{
  int      ierr;
  PC_Shell *shell;

  PetscFunctionBegin;
  pc->destroy    = PCDestroy_Shell;
  shell          = PetscNew(PC_Shell); CHKPTRQ(shell);
  PLogObjectMemory(pc,sizeof(PC_Shell));

  pc->data         = (void *) shell;
  pc->apply        = PCApply_Shell;
  pc->applyrich    = 0;
  pc->setup        = 0;
  pc->view         = PCView_Shell;
  pc->name         = 0;
  shell->apply     = 0;
  shell->name      = 0;
  shell->applyrich = 0;
  shell->ctxrich   = 0;
  shell->ctx       = 0;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApply_C","PCShellSetApply_Shell",
                    (void*)PCShellSetApply_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetName_C","PCShellSetName_Shell",
                    (void*)PCShellSetName_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellGetName_C","PCShellGetName_Shell",
                    (void*)PCShellGetName_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCShellSetApplyRichardson_C",
                    "PCShellSetApplyRichardson_Shell",
                    (void*)PCShellSetApplyRichardson_Shell);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}





