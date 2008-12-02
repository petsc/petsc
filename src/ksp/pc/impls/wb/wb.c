#define PETSCKSP_DLL


#include "petscpc.h"   /*I "petscpc.h" I*/
#include "petscmg.h"   /*I "petscmg.h" I*/
#include "petscda.h"   /*I "petscda.h" I*/
#include "../src/ksp/pc/impls/mg/mgimpl.h"

const char *PCExoticTypes[] = {"face","wirebasket","PCExoticType","PC_Exotic",0};

extern PetscErrorCode DAGetWireBasketInterpolation(DA,Mat,Mat*);
extern PetscErrorCode DAGetFaceInterpolation(DA,Mat,Mat*);

typedef struct {
  DA           da;
  PCExoticType type;
} PC_Exotic;

#undef __FUNCT__  
#define __FUNCT__ "PCExoticSetType"
/*@
   PCExoticSetType - Sets the type of coarse grid interpolation to use

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - either PC_EXOTIC_FACE or PC_EXOTIC_WIREBASKET (defaults to face)

   Level: intermediate


.seealso: PCEXOTIC, PCExoticType()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCExoticSetType(PC pc,PCExoticType type)
{
  PetscErrorCode ierr,(*f)(PC,PCExoticType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCExoticSetType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,type);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCExoticSetType_Exotic"
PetscErrorCode PETSCKSP_DLLEXPORT PCExoticSetType_Exotic(PC pc,PCExoticType type)
{
  PC_MG     **mg = (PC_MG**)pc->data;
  PC_Exotic *ctx = (PC_Exotic*) mg[0]->innerctx;

  PetscFunctionBegin;
  ctx->type = type;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_Exotic"
PetscErrorCode PCSetUp_Exotic(PC pc)
{
  PetscErrorCode ierr;
  Mat            A,P;
  PC_MG          **mg = (PC_MG**)pc->data;
  PC_Exotic      *ex = (PC_Exotic*) mg[0]->innerctx;
  DA             da = ex->da;

  PetscFunctionBegin;
  ierr = PCGetOperators(pc,PETSC_NULL,&A,PETSC_NULL);CHKERRQ(ierr);
  if (ex->type == PC_EXOTIC_FACE) {
    ierr = DAGetFaceInterpolation(da,A,&P);CHKERRQ(ierr);
  } else if (ex->type == PC_EXOTIC_WIREBASKET) {
    ierr = DAGetWireBasketInterpolation(da,A,&P);CHKERRQ(ierr);
  } else SETERRQ1(PETSC_ERR_PLIB,"Unknown exotic coarse space %d",ex->type);
  ierr = PCMGSetInterpolation(pc,1,P);CHKERRQ(ierr);
  ierr = MatDestroy(P);CHKERRQ(ierr);
  ierr = PCSetUp_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCDestroy_Exotic"
PetscErrorCode PCDestroy_Exotic(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          **mg = (PC_MG**)pc->data;
  PC_Exotic      *ctx = (PC_Exotic*) mg[0]->innerctx;

  PetscFunctionBegin;
  ierr = DADestroy(ctx->da);CHKERRQ(ierr);
  ierr = PetscFree(ctx);CHKERRQ(ierr);
  ierr = PCDestroy_MG(pc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCSetUp_Exotic_Error"
PetscErrorCode PCSetUp_Exotic_Error(PC pc)
{
  PetscFunctionBegin;
  SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"You are using the Exotic preconditioner but never called PCExoticSetDA()");
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCExoticSetDA"
/*@
   PCExoticSetDA - Sets the DA that is to be used by the PCEXOTIC preconditioner

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  da - the da

   Level: intermediate


.seealso: PCEXOTIC, PCExoticType()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCExoticSetDA(PC pc,DA da)
{
  PetscErrorCode ierr,(*f)(PC,DA);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidHeaderSpecific(da,DM_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCExoticSetDA_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,da);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}


#undef __FUNCT__  
#define __FUNCT__ "PCExoticSetDA_Exotic"
PetscErrorCode PETSCKSP_DLLEXPORT PCExoticSetDA_Exotic(PC pc,DA da)
{
  PetscErrorCode ierr;
  PC_MG          **mg = (PC_MG**)pc->data;
  PC_Exotic      *ctx = (PC_Exotic*) mg[0]->innerctx;

  PetscFunctionBegin;
  ctx->da = da;
  pc->ops->setup = PCSetUp_Exotic;
  ierr   = PetscObjectReference((PetscObject)da);CHKERRQ(ierr); 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_Exotic"
PetscErrorCode PCView_Exotic(PC pc,PetscViewer viewer)
{
  PC_MG          **mg = (PC_MG**)pc->data;
  PetscErrorCode ierr;
  PetscTruth     iascii;
  PC_Exotic      *ctx = (PC_Exotic*) mg[0]->innerctx;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"    Exotic type = %s\n",PCExoticTypes[ctx->type]);CHKERRQ(ierr);
  }
  ierr = PCView_MG(pc,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Exotic"
PetscErrorCode PCSetFromOptions_Exotic(PC pc)
{
  PetscErrorCode ierr;
  PetscTruth     flg;
  PC_MG          **mg = (PC_MG**)pc->data;
  PCExoticType   mgctype;
  PC_Exotic      *ctx = (PC_Exotic*) mg[0]->innerctx;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Exotic coarse space options");CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-pc_exotic_type","face or wirebasket","PCExoticSetType",PCExoticTypes,(PetscEnum)ctx->type,(PetscEnum*)&mgctype,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCExoticSetType(pc,mgctype);CHKERRQ(ierr);
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/*MC
     PCEXOTIC - Two level overlapping Schwarz preconditioner with exotic (non-standard) coarse grid spaces

     This uses the PCMG infrastructure restricted to two levels and the face and wirebasket based coarse
   grid spaces. These coarse grid spaces originate in the work of Bramble, Pasiak (Sp) and Schatz, they
   were generalized slightly in "Domain Decomposition Method for Linear Elasticity", Ph. D. thesis, Barry Smith,
   New York University, 1990. These were developed in the context of iterative substructuring preconditioners.
   They were then ingeniously applied as coarse grid spaces for overlapping Schwarz methods by XXXX and Widlund.
   They refer to them as GDSW (generalized Dryja, Smith, Widlund preconditioners).

   Options Database: The usual PCMG options are supported, such as -mg_levels_pc_type <type> -mg_coarse_pc_type <type>
      -pc_mg_type <type>

.seealso:  PCMG, PCExoticSetDA(), PCExoticType, PCExoticSetType()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Exotic"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Exotic(PC pc)
{
  PetscErrorCode ierr;
  PC_Exotic      *ex;
  PC_MG          **mg;

  PetscFunctionBegin;
  ierr = PCSetType(pc,PCMG);CHKERRQ(ierr);
  ierr = PCMGSetLevels(pc,2,PETSC_NULL);CHKERRQ(ierr);
  ierr = PCMGSetGalerkin(pc);CHKERRQ(ierr);
  ierr = PetscNew(PC_Exotic,&ex);CHKERRQ(ierr);
  ex->type = PC_EXOTIC_FACE;
  mg = (void*)(PC_MG**)pc->data;
  mg[0]->innerctx = ex;


  pc->ops->setfromoptions = PCSetFromOptions_Exotic;
  pc->ops->view           = PCView_Exotic;
  pc->ops->destroy        = PCDestroy_Exotic;
  pc->ops->setup          = PCSetUp_Exotic_Error;
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCExoticSetType_C","PCExoticSetType_Exotic",PCExoticSetType_Exotic);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCExoticSetDA_C","PCExoticSetDA_Exotic",PCExoticSetDA_Exotic);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END
