/*$Id: pcsles.c,v 1.39 2001/04/10 19:36:17 bsmith Exp $*/


#include "src/sles/pc/pcimpl.h"   /*I "petscpc.h" I*/
#include "petscsles.h"            /*I "petscsles.h" I*/

typedef struct {
  PetscTruth use_true_matrix;       /* use mat rather than pmat in inner linear solve */
  SLES       sles; 
  int        its;                   /* total number of iterations SLES uses */
} PC_SLES;

#undef __FUNCT__  
#define __FUNCT__ "PCApply_SLES"
static int PCApply_SLES(PC pc,Vec x,Vec y)
{
  int     ierr,its;
  PC_SLES *jac = (PC_SLES*)pc->data;
  KSP     ksp;

  PetscFunctionBegin;
  ierr      = SLESSolve(jac->sles,x,y);CHKERRQ(ierr);
  ierr      = SLESGetKSP(jac->sles,&ksp);CHKERRQ(ierr);
  ierr      = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  jac->its += its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApplyTranspose_SLES"
static int PCApplyTranspose_SLES(PC pc,Vec x,Vec y)
{
  int     its,ierr;
  PC_SLES *jac = (PC_SLES*)pc->data;
  KSP     ksp;

  PetscFunctionBegin;
  ierr      = SLESSolveTranspose(jac->sles,x,y);CHKERRQ(ierr);
  ierr      = SLESGetKSP(jac->sles,&ksp);CHKERRQ(ierr);
  ierr      = KSPGetIterationNumber(ksp,&its);CHKERRQ(ierr);
  jac->its += its;
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_SLES"
static int PCSetUp_SLES(PC pc)
{
  int     ierr;
  PC_SLES *jac = (PC_SLES*)pc->data;
  Mat     mat;

  PetscFunctionBegin;
  ierr = SLESSetFromOptions(jac->sles);CHKERRQ(ierr);
  if (jac->use_true_matrix) mat = pc->mat;
  else                      mat = pc->pmat;

  ierr = SLESSetOperators(jac->sles,mat,pc->pmat,pc->flag);CHKERRQ(ierr);
  ierr = SLESSetUp(jac->sles,pc->vec,pc->vec);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Default destroy, if it has never been setup */
#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_SLES"
static int PCDestroy_SLES(PC pc)
{
  PC_SLES *jac = (PC_SLES*)pc->data;
  int     ierr;

  PetscFunctionBegin;
  ierr = SLESDestroy(jac->sles);CHKERRQ(ierr);
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_SLES"
static int PCView_SLES(PC pc,PetscViewer viewer)
{
  PC_SLES    *jac = (PC_SLES*)pc->data;
  int        ierr;
  PetscTruth isascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&isascii);CHKERRQ(ierr);
  if (isascii) {
    if (jac->use_true_matrix) {
      ierr = PetscViewerASCIIPrintf(viewer,"Using true matrix (not preconditioner matrix) on inner solve\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"KSP and PC on SLES preconditioner follow\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"---------------------------------\n");CHKERRQ(ierr);
  } else {
    SETERRQ1(1,"Viewer type %s not supported for this object",((PetscObject)viewer)->type_name);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = SLESView(jac->sles,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  if (isascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"---------------------------------\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_SLES"
static int PCSetFromOptions_SLES(PC pc){
  int        ierr;
  PetscTruth flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("SLES preconditioner options");CHKERRQ(ierr);
    ierr = PetscOptionsName("-pc_sles_true","Use true matrix to define inner linear system, not preconditioner matrix","PCSLESSetUseTrue",&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCSLESSetUseTrue(pc);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCSLESSetUseTrue_SLES"
int PCSLESSetUseTrue_SLES(PC pc)
{
  PC_SLES   *jac;

  PetscFunctionBegin;
  jac                  = (PC_SLES*)pc->data;
  jac->use_true_matrix = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCSLESGetSLES_SLES"
int PCSLESGetSLES_SLES(PC pc,SLES *sles)
{
  PC_SLES   *jac;

  PetscFunctionBegin;
  jac          = (PC_SLES*)pc->data;
  *sles        = jac->sles;
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCSLESSetUseTrue"
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
  int ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCSLESSetUseTrue_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSLESGetSLES"
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
  int ierr,(*f)(PC,SLES*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE);
  if (!pc->setupcalled) SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Must call SLESSetUp first");
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCSLESGetSLES_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,sles);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

/*MC
     PCSLES -    Defines a preconditioner that can consist of any SLES solver.
                 This allows, for example, embedding a Krylov method inside a preconditioner.

   Options Database Key:
.     -pc_sles_true - use the matrix that defines the linear system as the matrix for the
                    inner solver, otherwise by default it uses the matrix used to construct
                    the preconditioner (see PCSetOperators())

   Level: intermediate

   Concepts: inner iteration

   Notes: Using a Krylov method inside another Krylov method can be dangerous (you get divergence or
          the incorrect answer) unless you use KSPFGMRES as the other Krylov method


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCSHELL, PCCOMPOSITE, PCSLESUseTrue(), PCSLESGetSLES()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_SLES"
int PCCreate_SLES(PC pc)
{
  int       ierr;
  char      *prefix;
  PC_SLES   *jac;

  PetscFunctionBegin;
  ierr = PetscNew(PC_SLES,&jac);CHKERRQ(ierr);
  PetscLogObjectMemory(pc,sizeof(PC_SLES));
  pc->ops->apply              = PCApply_SLES;
  pc->ops->applytranspose     = PCApplyTranspose_SLES;
  pc->ops->setup              = PCSetUp_SLES;
  pc->ops->destroy            = PCDestroy_SLES;
  pc->ops->setfromoptions     = PCSetFromOptions_SLES;
  pc->ops->view               = PCView_SLES;
  pc->ops->applyrichardson    = 0;

  pc->data               = (void*)jac;
  ierr                   = SLESCreate(pc->comm,&jac->sles);CHKERRQ(ierr);

  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = SLESSetOptionsPrefix(jac->sles,prefix);CHKERRQ(ierr);
  ierr = SLESAppendOptionsPrefix(jac->sles,"sles_");CHKERRQ(ierr);
  jac->use_true_matrix = PETSC_FALSE;
  jac->its             = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCSLESSetUseTrue_C","PCSLESSetUseTrue_SLES",
                    PCSLESSetUseTrue_SLES);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCSLESGetSLES_C","PCSLESGetSLES_SLES",
                    PCSLESGetSLES_SLES);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

