/*$Id: pcsles.c,v 1.26 1999/11/05 14:46:29 bsmith Exp bsmith $*/
/*
      Defines a preconditioner that can consist of any SLES solver.
    This allows embedding a Krylov method inside a preconditioner.
*/
#include "src/sles/pc/pcimpl.h"   /*I "pc.h" I*/
#include "sles.h"            /*I "sles.h" I*/

typedef struct {
  PetscTruth use_true_matrix;       /* use mat rather than pmat in inner linear solve */
  SLES       sles; 
  int        its;                   /* total number of iterations SLES uses */
} PC_SLES;

#undef __FUNC__  
#define __FUNC__ "PCApply_SLES"
static int PCApply_SLES(PC pc,Vec x,Vec y)
{
  int     ierr,its;
  PC_SLES *jac = (PC_SLES *) pc->data;

  PetscFunctionBegin;
  ierr      = SLESSolve(jac->sles,x,y,&its);CHKERRQ(ierr);
  jac->its += its;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCApplyTranspose_SLES"
static int PCApplyTranspose_SLES(PC pc,Vec x,Vec y)
{
  int     ierr,its;
  PC_SLES *jac = (PC_SLES *) pc->data;

  PetscFunctionBegin;
  ierr      = SLESSolveTranspose(jac->sles,x,y,&its);CHKERRQ(ierr);
  jac->its += its;
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetUp_SLES"
static int PCSetUp_SLES(PC pc)
{
  int     ierr;
  PC_SLES *jac = (PC_SLES *) pc->data;
  Mat     mat;

  PetscFunctionBegin;
  if (jac->use_true_matrix) mat = pc->mat;
  else                      mat = pc->pmat;

  ierr = SLESSetOperators(jac->sles,mat,pc->pmat,pc->flag);CHKERRQ(ierr);
  ierr = SLESSetUp(jac->sles,pc->vec,pc->vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Default destroy, if it has never been setup */
#undef __FUNC__  
#define __FUNC__ "PCDestroy_SLES"
static int PCDestroy_SLES(PC pc)
{
  PC_SLES *jac = (PC_SLES *) pc->data;
  int     ierr;

  PetscFunctionBegin;
  ierr = SLESDestroy(jac->sles);CHKERRQ(ierr);
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


#undef __FUNC__  
#define __FUNC__ "PCPrintHelp_SLES"
static int PCPrintHelp_SLES(PC pc,char *p)
{
  int ierr;

  PetscFunctionBegin;
  ierr = (*PetscHelpPrintf)(pc->comm," Options for PCSLES preconditioner:\n");CHKERRQ(ierr);
  ierr = (*PetscHelpPrintf)(pc->comm," %ssles : prefix to control options for individual blocks.\
 Add before the \n      usual KSP and PC option names (e.g., %ssles_ksp_type\
 <ksptype>)\n",p,p);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCView_SLES"
static int PCView_SLES(PC pc,Viewer viewer)
{
  PC_SLES    *jac = (PC_SLES *) pc->data;
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,ASCII_VIEWER,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (jac->use_true_matrix) {
      ierr = ViewerASCIIPrintf(viewer,"Using true matrix (not preconditioner matrix) on inner solve\n");CHKERRQ(ierr);
    }
    ierr = ViewerASCIIPrintf(viewer,"KSP and PC on SLES preconditioner follow\n");CHKERRQ(ierr);
    ierr = ViewerASCIIPrintf(viewer,"---------------------------------\n");CHKERRQ(ierr);
  } else {
    SETERRQ1(1,1,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
  }
  ierr = ViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = SLESView(jac->sles,viewer);CHKERRQ(ierr);
  ierr = ViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  if (isascii) {
    ierr = ViewerASCIIPrintf(viewer,"---------------------------------\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSetFromOptions_SLES"
static int PCSetFromOptions_SLES(PC pc)
{
  PC_SLES    *jac = (PC_SLES *) pc->data;
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = OptionsHasName(pc->prefix,"-pc_sles_true",&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCSLESSetUseTrue(pc);CHKERRQ(ierr);
  }
  ierr = SLESSetFromOptions(jac->sles);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCSLESSetUseTrue_SLES"
int PCSLESSetUseTrue_SLES(PC pc)
{
  PC_SLES   *jac;

  PetscFunctionBegin;
  jac                  = (PC_SLES *) pc->data;
  jac->use_true_matrix = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCSLESGetSLES_SLES"
int PCSLESGetSLES_SLES(PC pc,SLES *sles)
{
  PC_SLES   *jac;

  PetscFunctionBegin;
  jac          = (PC_SLES *) pc->data;
  *sles        = jac->sles;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNC__  
#define __FUNC__ "PCSLESSetUseTrue"
/*@
   PCSLESSetUseTrue - Sets a flag to indicate that the true matrix (rather than
   the matrix used to define the preconditioner) is used to compute the
   residual inside the inner solve.

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_sles_true - Activates PCSLESSetUseTrue()

   Note:
   For the common case in which the preconditioning and linear 
   system matrices are identical, this routine is unnecessary.

   Level: advanced

.keywords:  PC, SLES, set, true, local, flag

.seealso: PCSetOperators(), PCBJacobiSetUseTrueLocal()
@*/
int PCSLESSetUseTrue(PC pc)
{
  int ierr, (*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCSLESSetUseTrue_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNC__  
#define __FUNC__ "PCSLESGetSLES"
/*@C
   PCSLESGetSLES - Gets the SLES context for a SLES PC.

   Not Collective but SLES returned is parallel if PC was parallel

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  sles - the PC solver

   Notes:
   You must call SLESSetUp() before calling PCSLESGetSLES().

   Level: advanced

.keywords:  PC, SLES, get, context
@*/
int PCSLESGetSLES(PC pc,SLES *sles)
{
  int ierr, (*f)(PC,SLES*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (!pc->setupcalled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,0,"Must call SLESSetUp first");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCSLESGetSLES_C",(void **)&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,sles);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNC__  
#define __FUNC__ "PCCreate_SLES"
int PCCreate_SLES(PC pc)
{
  int       ierr;
  char      *prefix;
  PC_SLES   *jac = PetscNew(PC_SLES);CHKPTRQ(jac);

  PetscFunctionBegin;
  PLogObjectMemory(pc,sizeof(PC_SLES));
  pc->ops->apply              = PCApply_SLES;
  pc->ops->applytranspose     = PCApplyTranspose_SLES;
  pc->ops->setup              = PCSetUp_SLES;
  pc->ops->destroy            = PCDestroy_SLES;
  pc->ops->setfromoptions     = PCSetFromOptions_SLES;
  pc->ops->printhelp          = PCPrintHelp_SLES;
  pc->ops->view               = PCView_SLES;
  pc->ops->applyrichardson    = 0;

  pc->data               = (void *) jac;
  ierr                   = SLESCreate(pc->comm,&jac->sles);CHKERRQ(ierr);

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = SLESSetOptionsPrefix(jac->sles,prefix);CHKERRQ(ierr);
  ierr = SLESAppendOptionsPrefix(jac->sles,"sles_");CHKERRQ(ierr);
  jac->use_true_matrix = PETSC_FALSE;
  jac->its             = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCSLESSetUseTrue_C","PCSLESSetUseTrue_SLES",
                    (void*)PCSLESSetUseTrue_SLES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCSLESGetSLES_C","PCSLESGetSLES_SLES",
                    (void*)PCSLESGetSLES_SLES);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

