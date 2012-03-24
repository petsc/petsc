
/*
      Defines a preconditioner defined by R^T S R 
*/
#include <petsc-private/pcimpl.h>   /*I "petscpc.h" I*/
#include <petscksp.h>         /*I "petscksp.h" I*/

typedef struct {
  KSP  ksp;
  Mat  R,P;
  Vec  b,x;
} PC_Galerkin;

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Galerkin"
static PetscErrorCode PCApply_Galerkin(PC pc,Vec x,Vec y)
{
  PetscErrorCode   ierr;
  PC_Galerkin     *jac = (PC_Galerkin*)pc->data;

  PetscFunctionBegin;
  if (jac->R) {ierr = MatRestrict(jac->R,x,jac->b);CHKERRQ(ierr);}
  else {ierr = MatRestrict(jac->P,x,jac->b);CHKERRQ(ierr);}
  ierr = KSPSolve(jac->ksp,jac->b,jac->x);CHKERRQ(ierr);
  if (jac->P) {ierr = MatInterpolate(jac->P,jac->x,y);CHKERRQ(ierr);}
  else {ierr = MatInterpolate(jac->R,jac->x,y);CHKERRQ(ierr);}
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Galerkin"
static PetscErrorCode PCSetUp_Galerkin(PC pc)
{
  PetscErrorCode  ierr;
  PC_Galerkin     *jac = (PC_Galerkin*)pc->data;
  PetscBool       a;
  Vec             *xx,*yy;

  PetscFunctionBegin;
  if (!jac->x) {
    ierr = KSPGetOperatorsSet(jac->ksp,&a,PETSC_NULL);CHKERRQ(ierr);
    if (!a) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set operator of PCGALERKIN KSP with PCGalerkinGetKSP()/KSPSetOperators()");
    ierr   = KSPGetVecs(jac->ksp,1,&xx,1,&yy);CHKERRQ(ierr);    
    jac->x = *xx;
    jac->b = *yy;
    ierr   = PetscFree(xx);CHKERRQ(ierr);
    ierr   = PetscFree(yy);CHKERRQ(ierr);
  }
  if (!jac->R && !jac->P) SETERRQ(((PetscObject)pc)->comm,PETSC_ERR_ARG_WRONGSTATE,"Must set restriction or interpolation of PCGALERKIN with PCGalerkinSetRestriction/Interpolation()");
  /* should check here that sizes of R/P match size of a */
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCReset_Galerkin"
static PetscErrorCode PCReset_Galerkin(PC pc)
{
  PC_Galerkin     *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = MatDestroy(&jac->R);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->P);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->x);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->b);CHKERRQ(ierr);
  ierr = KSPReset(jac->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Galerkin"
static PetscErrorCode PCDestroy_Galerkin(PC pc)
{
  PC_Galerkin     *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PCReset_Galerkin(pc);CHKERRQ(ierr);
  ierr = KSPDestroy(&jac->ksp);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_Galerkin"
static PetscErrorCode PCView_Galerkin(PC pc,PetscViewer viewer)
{
  PC_Galerkin     *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode   ierr;
  PetscBool        iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Galerkin PC\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"KSP on Galerkin follow\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"---------------------------------\n");CHKERRQ(ierr);
  } else {
    SETERRQ1(((PetscObject)pc)->comm,PETSC_ERR_SUP,"Viewer type %s not supported for PCGalerkin",((PetscObject)viewer)->type_name);
  }
  ierr = KSPView(jac->ksp,viewer);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGalerkinGetKSP_Galerkin"
PetscErrorCode  PCGalerkinGetKSP_Galerkin(PC pc,KSP *ksp)
{
  PC_Galerkin     *jac = (PC_Galerkin*)pc->data;

  PetscFunctionBegin;
  *ksp = jac->ksp;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGalerkinSetRestriction_Galerkin"
PetscErrorCode  PCGalerkinSetRestriction_Galerkin(PC pc,Mat R)
{
  PC_Galerkin     *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)R);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->R);CHKERRQ(ierr);
  jac->R = R;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCGalerkinSetInterpolation_Galerkin"
PetscErrorCode  PCGalerkinSetInterpolation_Galerkin(PC pc,Mat P)
{
  PC_Galerkin     *jac = (PC_Galerkin*)pc->data;
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)P);CHKERRQ(ierr);
  ierr = MatDestroy(&jac->P);CHKERRQ(ierr);
  jac->P = P;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCGalerkinSetRestriction"
/*@
   PCGalerkinSetRestriction - Sets the restriction operator for the "Galerkin-type" preconditioner
   
   Logically Collective on PC

   Input Parameter:
+  pc - the preconditioner context
-  R - the restriction operator

   Notes: Either this or PCGalerkinSetInterpolation() or both must be called

   Level: Intermediate

.keywords: PC, set, Galerkin preconditioner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCGALERKIN,
           PCGalerkinSetInterpolation(), PCGalerkinGetKSP()

@*/
PetscErrorCode  PCGalerkinSetRestriction(PC pc,Mat R)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGalerkinSetRestriction_C",(PC,Mat),(pc,R));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGalerkinSetRestriction"
/*@
   PCGalerkinSetInterpolation - Sets the interpolation operator for the "Galerkin-type" preconditioner
   
   Logically Collective on PC

   Input Parameter:
+  pc - the preconditioner context
-  R - the interpolation operator

   Notes: Either this or PCGalerkinSetRestriction() or both must be called

   Level: Intermediate

.keywords: PC, set, Galerkin preconditioner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCGALERKIN,
           PCGalerkinSetRestriction(), PCGalerkinGetKSP()

@*/
PetscErrorCode  PCGalerkinSetInterpolation(PC pc,Mat P)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCGalerkinSetInterpolation_C",(PC,Mat),(pc,P));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCGalerkinGetKSP"
/*@
   PCGalerkinGetKSP - Gets the KSP object in the Galerkin PC.
   
   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  ksp - the KSP object

   Level: Intermediate

.keywords: PC, get, Galerkin preconditioner, sub preconditioner

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PCGALERKIN,
           PCGalerkinSetRestriction(), PCGalerkinSetInterpolation()

@*/
PetscErrorCode  PCGalerkinGetKSP(PC pc,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(ksp,2);
  ierr = PetscUseMethod(pc,"PCGalerkinGetKSP_C",(PC,KSP *),(pc,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/* -------------------------------------------------------------------------------------------*/

/*MC
     PCGALERKIN - Build (part of) a preconditioner by P S R (where P is often R^T)

$   Use PCGalerkinSetRestriction(pc,R) and/or PCGalerkinSetInterpolation(pc,P) followed by 
$   PCGalerkinGetKSP(pc,&ksp); KSPSetOperators(ksp,A,....)

   Level: intermediate

   Developer Note: If KSPSetOperators() has not been called then PCGALERKIN could use MatRARt() or MatPtAP() to compute
                   the operators automatically.
                   Should there be a prefix for the inner KSP.
                   There is no KSPSetFromOptions_Galerkin() that calls KSPSetFromOptions() on the inner KSP

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCSHELL, PCKSP, PCGalerkinSetRestriction(), PCGalerkinSetInterpolation(), PCGalerkinGetKSP()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Galerkin"
PetscErrorCode  PCCreate_Galerkin(PC pc)
{
  PetscErrorCode ierr;
  PC_Galerkin    *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_Galerkin,&jac);CHKERRQ(ierr);
  pc->ops->apply              = PCApply_Galerkin;
  pc->ops->setup              = PCSetUp_Galerkin;
  pc->ops->reset              = PCReset_Galerkin;
  pc->ops->destroy            = PCDestroy_Galerkin;
  pc->ops->view               = PCView_Galerkin;
  pc->ops->applyrichardson    = 0;

  ierr = KSPCreate(((PetscObject)pc)->comm,&jac->ksp);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)jac->ksp,(PetscObject)pc,1);CHKERRQ(ierr);

  pc->data               = (void*)jac;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCGalerkinSetRestriction_C","PCGalerkinSetRestriction_Galerkin",
                    PCGalerkinSetRestriction_Galerkin);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCGalerkinSetInterpolation_C","PCGalerkinSetInterpolation_Galerkin",
                    PCGalerkinSetInterpolation_Galerkin);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCGalerkinGetKSP_C","PCGalerkinGetKSP_Galerkin",
                    PCGalerkinGetKSP_Galerkin);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

