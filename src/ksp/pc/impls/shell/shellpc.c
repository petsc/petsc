/*$Id: shellpc.c,v 1.77 2001/08/21 21:03:18 bsmith Exp $*/

/*
   This provides a simple shell for Fortran (and C programmers) to 
  create their own preconditioner without writing much interface code.
*/

#include "src/ksp/pc/pcimpl.h"        /*I "petscpc.h" I*/
#include "src/vec/vecimpl.h"  

typedef struct {
  void *ctx,*ctxrich;    /* user provided contexts for preconditioner */
  int  (*setup)(void *);
  int  (*apply)(void *,Vec,Vec);
  int  (*view)(void *,PetscViewer);
  int  (*applytranspose)(void *,Vec,Vec);
  int  (*applyrich)(void *,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int);
  char *name;
} PC_Shell;

#undef __FUNCT__  
#define __FUNCT__ "PCApply_SetUp"
static int PCSetUp_Shell(PC pc)
{
  PC_Shell *shell;
  int      ierr;

  PetscFunctionBegin;
  shell = (PC_Shell*)pc->data;
  if (shell->setup) {
    ierr  = (*shell->setup)(shell->ctx);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Shell"
static int PCApply_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell *shell;
  int      ierr;

  PetscFunctionBegin;
  shell = (PC_Shell*)pc->data;
  if (!shell->apply) SETERRQ(1,"No apply() routine provided to Shell PC");
  ierr  = (*shell->apply)(shell->ctx,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_Shell"
static int PCApplyTranspose_Shell(PC pc,Vec x,Vec y)
{
  PC_Shell *shell;
  int      ierr;

  PetscFunctionBegin;
  shell = (PC_Shell*)pc->data;
  if (!shell->applytranspose) SETERRQ(1,"No applytranspose() routine provided to Shell PC");
  ierr  = (*shell->applytranspose)(shell->ctx,x,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyRichardson_Shell"
static int PCApplyRichardson_Shell(PC pc,Vec x,Vec y,Vec w,PetscReal rtol,PetscReal atol, PetscReal dtol,int it)
{
  int      ierr;
  PC_Shell *shell;

  PetscFunctionBegin;
  shell = (PC_Shell*)pc->data;
  ierr  = (*shell->applyrich)(shell->ctxrich,x,y,w,rtol,atol,dtol,it);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Shell"
static int PCDestroy_Shell(PC pc)
{
  PC_Shell *shell = (PC_Shell*)pc->data;
  int      ierr;

  PetscFunctionBegin;
  if (shell->name) {ierr = PetscFree(shell->name);}
  ierr = PetscFree(shell);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_Shell"
static int PCView_Shell(PC pc,PetscViewer viewer)
{
  PC_Shell   *shell = (PC_Shell*)pc->data;
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (shell->name) {ierr = PetscViewerASCIIPrintf(viewer,"  Shell: %s\n",shell->name);CHKERRQ(ierr);}
    else             {ierr = PetscViewerASCIIPrintf(viewer,"  Shell: no name\n");CHKERRQ(ierr);}
  }
  if (shell->view) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr  = (*shell->view)(shell->ctx,viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/
EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCShellSetSetUp_Shell"
int PCShellSetSetUp_Shell(PC pc, int (*setup)(void*))
{
  PC_Shell *shell;

  PetscFunctionBegin;
  shell        = (PC_Shell*)pc->data;
  shell->setup = setup;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCShellSetApply_Shell"
int PCShellSetApply_Shell(PC pc,int (*apply)(void*,Vec,Vec),void *ptr)
{
  PC_Shell *shell;

  PetscFunctionBegin;
  shell        = (PC_Shell*)pc->data;
  shell->apply = apply;
  shell->ctx   = ptr;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCShellSetView_Shell"
int PCShellSetView_Shell(PC pc,int (*view)(void*,PetscViewer))
{
  PC_Shell *shell;

  PetscFunctionBegin;
  shell        = (PC_Shell*)pc->data;
  shell->view = view;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCShellSetApplyTranspose_Shell"
int PCShellSetApplyTranspose_Shell(PC pc,int (*applytranspose)(void*,Vec,Vec))
{
  PC_Shell *shell;

  PetscFunctionBegin;
  shell                 = (PC_Shell*)pc->data;
  shell->applytranspose = applytranspose;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCShellSetName_Shell"
int PCShellSetName_Shell(PC pc,const char name[])
{
  PC_Shell *shell;
  int      ierr;

  PetscFunctionBegin;
  shell = (PC_Shell*)pc->data;
  ierr  = PetscStrallocpy(name,&shell->name);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCShellGetName_Shell"
int PCShellGetName_Shell(PC pc,char *name[])
{
  PC_Shell *shell;

  PetscFunctionBegin;
  shell  = (PC_Shell*)pc->data;
  *name  = shell->name;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCShellSetApplyRichardson_Shell"
int PCShellSetApplyRichardson_Shell(PC pc,int (*apply)(void*,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int),void *ptr)
{
  PC_Shell *shell;

  PetscFunctionBegin;
  shell                     = (PC_Shell*)pc->data;
  pc->ops->applyrichardson  = PCApplyRichardson_Shell;
  shell->applyrich          = apply;
  shell->ctxrich            = ptr;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------------*/

#undef __FUNCT__  
#define __FUNCT__ "PCShellSetSetUp"
/*@C
   PCShellSetSetUp - Sets routine to use to "setup" the preconditioner whenever the 
   matrix operator is changed.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
.  setup - the application-provided setup routine

   Calling sequence of setup:
.vb
   int setup (void *ptr)
.ve

.  ptr - the application context

   Level: developer

.keywords: PC, shell, set, setup, user-provided

.seealso: PCShellSetApplyRichardson(), PCShellSetApply()
@*/
int PCShellSetSetUp(PC pc,int (*setup)(void*))
{
  int ierr,(*f)(PC,int (*)(void*));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellSetSetUp_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,setup);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PCShellSetView"
/*@C
   PCShellSetView - Sets routine to use as viewer of shell preconditioner

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  view - the application-provided view routine

   Calling sequence of apply:
.vb
   int view(void *ptr,PetscViewer v)
.ve

+  ptr - the application context
-  v   - viewer

   Level: developer

.keywords: PC, shell, set, apply, user-provided

.seealso: PCShellSetApplyRichardson(), PCShellSetSetUp(), PCShellSetApplyTranspose()
@*/
int PCShellSetView(PC pc,int (*view)(void*,PetscViewer))
{
  int ierr,(*f)(PC,int (*)(void*,PetscViewer));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellSetView_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,view);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCShellSetApply"
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

+  ptr - the application context
.  xin - input vector
-  xout - output vector

   Level: developer

.keywords: PC, shell, set, apply, user-provided

.seealso: PCShellSetApplyRichardson(), PCShellSetSetUp(), PCShellSetApplyTranspose()
@*/
int PCShellSetApply(PC pc,int (*apply)(void*,Vec,Vec),void *ptr)
{
  int ierr,(*f)(PC,int (*)(void*,Vec,Vec),void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellSetApply_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,apply,ptr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCShellSetApplyTranspose"
/*@C
   PCShellSetApplyTranspose - Sets routine to use as preconditioner transpose.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  apply - the application-provided preconditioning transpose routine

   Calling sequence of apply:
.vb
   int applytranspose (void *ptr,Vec xin,Vec xout)
.ve

+  ptr - the application context
.  xin - input vector
-  xout - output vector

   Level: developer

   Notes: 
   Uses the same context variable as PCShellSetApply().

.keywords: PC, shell, set, apply, user-provided

.seealso: PCShellSetApplyRichardson(), PCShellSetSetUp(), PCShellSetApply()
@*/
int PCShellSetApplyTranspose(PC pc,int (*applytranspose)(void*,Vec,Vec))
{
  int ierr,(*f)(PC,int (*)(void*,Vec,Vec));

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellSetApplyTranspose_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,applytranspose);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCShellSetName"
/*@C
   PCShellSetName - Sets an optional name to associate with a shell
   preconditioner.

   Not Collective

   Input Parameters:
+  pc - the preconditioner context
-  name - character string describing shell preconditioner

   Level: developer

.keywords: PC, shell, set, name, user-provided

.seealso: PCShellGetName()
@*/
int PCShellSetName(PC pc,const char name[])
{
  int ierr,(*f)(PC,const char []);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellSetName_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,name);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCShellGetName"
/*@C
   PCShellGetName - Gets an optional name that the user has set for a shell
   preconditioner.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  name - character string describing shell preconditioner (you should not free this)

   Level: developer

.keywords: PC, shell, get, name, user-provided

.seealso: PCShellSetName()
@*/
int PCShellGetName(PC pc,char *name[])
{
  int ierr,(*f)(PC,char *[]);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellGetName_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,name);CHKERRQ(ierr);
  } else {
    SETERRQ(1,"Not shell preconditioner, cannot get name");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCShellSetApplyRichardson"
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
   int apply (void *ptr,Vec b,Vec x,Vec r,PetscReal rtol,PetscReal atol,PetscReal dtol,int maxits)
.ve

+  ptr - the application context
.  b - right-hand-side
.  x - current iterate
.  r - work space
.  rtol - relative tolerance of residual norm to stop at
.  atol - absolute tolerance of residual norm to stop at
.  dtol - if residual norm increases by this factor than return
-  maxits - number of iterations to run

   Level: developer

.keywords: PC, shell, set, apply, Richardson, user-provided

.seealso: PCShellSetApply()
@*/
int PCShellSetApplyRichardson(PC pc,int (*apply)(void*,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int),void *ptr)
{
  int ierr,(*f)(PC,int (*)(void*,Vec,Vec,Vec,PetscReal,PetscReal,PetscReal,int),void *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCShellSetApplyRichardson_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,apply,ptr);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/*MC
   PCSHELL - Creates a new preconditioner class for use with your 
              own private data storage format.

   Level: advanced

   Concepts: providing your own preconditioner

  Usage:
$             int (*mult)(void *,Vec,Vec);
$             int (*setup)(void *);
$             PCCreate(comm,&pc);
$             PCSetType(pc,PCSHELL);
$             PCShellSetApply(pc,mult,ctx);
$             PCShellSetSetUp(pc,setup);       (optional)

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           KSPSHELL(), MATSHELL(), PCShellSetUp(), PCShellSetApply(), PCShellSetView(), 
           PCShellSetApplyTranpose(), PCShellSetName(), PCShellSetApplyRichardson(), 
           PCShellGetName()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Shell"
int PCCreate_Shell(PC pc)
{
  int      ierr;
  PC_Shell *shell;

  PetscFunctionBegin;
  pc->ops->destroy    = PCDestroy_Shell;
  ierr                = PetscNew(PC_Shell,&shell);CHKERRQ(ierr);
  PetscLogObjectMemory(pc,sizeof(PC_Shell));

  pc->data         = (void*)shell;
  pc->name         = 0;

  pc->ops->apply           = PCApply_Shell;
  pc->ops->view            = PCView_Shell;
  pc->ops->applytranspose  = PCApplyTranspose_Shell;
  pc->ops->applyrichardson = 0;
  pc->ops->setup           = PCSetUp_Shell;
  pc->ops->view            = PCView_Shell;

  shell->apply          = 0;
  shell->applytranspose = 0;
  shell->name           = 0;
  shell->applyrich      = 0;
  shell->ctxrich        = 0;
  shell->ctx            = 0;
  shell->setup          = 0;
  shell->view           = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCShellSetSetUp_C","PCShellSetSetUp_Shell",
                    PCShellSetSetUp_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCShellSetApply_C","PCShellSetApply_Shell",
                    PCShellSetApply_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCShellSetView_C","PCShellSetView_Shell",
                    PCShellSetView_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCShellSetApplyTranspose_C",
                    "PCShellSetApplyTranspose_Shell",
                    PCShellSetApplyTranspose_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCShellSetName_C","PCShellSetName_Shell",
                    PCShellSetName_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCShellGetName_C","PCShellGetName_Shell",
                    PCShellGetName_Shell);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCShellSetApplyRichardson_C",
                    "PCShellSetApplyRichardson_Shell",
                    PCShellSetApplyRichardson_Shell);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END






