#define PETSCSNES_DLL

#include "private/pcimpl.h"   /*I "petscpc.h" I*/
#include "petscdmmg.h"        /*I "petscdmmg.h" I*/

/* 
   Private context (data structure) for the DMMG preconditioner.  
*/
typedef struct {
  DMMG *dmmg;
} PC_DMMG;

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCDMMGSetDMMG_DMMG"
PetscErrorCode PETSCKSP_DLLEXPORT PCDMMGSetDMMG_DMMG(PC pc,DMMG *dmmg)
{
  PC_DMMG        *pcdmmg = (PC_DMMG*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  pcdmmg->dmmg = dmmg;
  PetscFunctionReturn(0);
}
EXTERN_C_END
#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_DMMG"
static PetscErrorCode PCSetUp_DMMG(PC pc)
{
  PC_DMMG        *pcdmmg = (PC_DMMG*)pc->data;
  DMMG           *dmmg = pcdmmg->dmmg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /* when used as preconditioner cannot provide right hand size (it is provided by PCApply()) */
  dmmg[dmmg[0]->nlevels-1]->rhs = PETSC_NULL;
  PetscFunctionReturn(0);
}
/* -------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCApply_DMMG"
static PetscErrorCode PCApply_DMMG(PC pc,Vec x,Vec y)
{
  PC_DMMG        *pcdmmg = (PC_DMMG*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = VecCopy(x,DMMGGetRHS(pcdmmg->dmmg));CHKERRQ(ierr);
  ierr = DMMGSolve(pcdmmg->dmmg);CHKERRQ(ierr);
  ierr = VecCopy(DMMGGetx(pcdmmg->dmmg),y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_DMMG"
static PetscErrorCode PCDestroy_DMMG(PC pc)
{
  PC_DMMG        *pcdmmg = (PC_DMMG*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (pcdmmg->dmmg) {ierr = DMMGDestroy(pcdmmg->dmmg);CHKERRQ(ierr);}
  ierr = PetscFree(pcdmmg);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_DMMG"
static PetscErrorCode PCSetFromOptions_DMMG(PC pc)
{
  PC_DMMG        *pcdmmg = (PC_DMMG*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("DMMG options");CHKERRQ(ierr);
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_DMMG"
static PetscErrorCode PCView_DMMG(PC pc,PetscViewer viewer)
{
  PC_DMMG        *pcdmmg = (PC_DMMG*)pc->data;
  PetscErrorCode ierr;
  PetscTruth     iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  DMMG based preconditioner: \n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
    ierr = DMMGView(pcdmmg->dmmg,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*MC
     PCDMMG - DMMG based preconditioner

   Level: Intermediate

  Concepts: DMMG, diagonal scaling, preconditioners


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCDMMGSetDMMG()
M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_DMMG"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_DMMG(PC pc)
{
  PC_DMMG        *pcdmmg;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  /*
     Creates the private data structure for this preconditioner and
     attach it to the PC object.
  */
  ierr          = PetscNew(PC_DMMG,&pcdmmg);CHKERRQ(ierr);
  pc->data      = (void*)pcdmmg;
  pcdmmg->dmmg  = 0;

  pc->ops->apply               = PCApply_DMMG;
  pc->ops->applytranspose      = PCApply_DMMG;
  pc->ops->setup               = PCSetUp_DMMG;
  pc->ops->destroy             = PCDestroy_DMMG;
  pc->ops->setfromoptions      = PCSetFromOptions_DMMG;
  pc->ops->view                = PCView_DMMG;
  pc->ops->applyrichardson     = 0;
  pc->ops->applysymmetricleft  = 0;
  pc->ops->applysymmetricright = 0;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCDMMGSetDMMG_C","PCDMMGSetDMMG_DMMG",PCDMMGSetDMMG_DMMG);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

#undef __FUNCT__  
#define __FUNCT__ "PCDMMGSetDMMG"
/*@
   PCDMMGSetDMMG - Sets the DMMG that is to be used to build the preconditioner

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  dmmg - the DMMG object

   Concepts: DMMG preconditioner

.seealso: PCDMMG
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCDMMGSetDMMG(PC pc,DMMG *dmmg)
{
  PetscErrorCode ierr,(*f)(PC,DMMG*);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCDMMGSetDMMG_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,dmmg);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}
