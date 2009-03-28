#define PETSCKSP_DLL

/*
      Defines a preconditioner that can consist of a collection of PCs
*/
#include "private/pcimpl.h"   /*I "petscpc.h" I*/
#include "petscksp.h"            /*I "petscksp.h" I*/

typedef struct _PC_CompositeLink *PC_CompositeLink;
struct _PC_CompositeLink {
  PC               pc;
  PC_CompositeLink next;
  PC_CompositeLink previous;
};
  
typedef struct {
  PC_CompositeLink head;
  PCCompositeType  type;
  Vec              work1;
  Vec              work2;
  PetscScalar      alpha;
  PetscTruth       use_true_matrix;
} PC_Composite;

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Composite_Multiplicative"
static PetscErrorCode PCApply_Composite_Multiplicative(PC pc,Vec x,Vec y)
{
  PetscErrorCode   ierr;
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;
  Mat              mat = pc->pmat;

  PetscFunctionBegin;
  if (!next) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPC() or -pc_composite_pcs");
  }
  if (next->next && !jac->work2) { /* allocate second work vector */
    ierr = VecDuplicate(jac->work1,&jac->work2);CHKERRQ(ierr);
  }
  ierr = PCApply(next->pc,x,y);CHKERRQ(ierr);
  if (jac->use_true_matrix) mat = pc->mat;
  while (next->next) {
    next = next->next;
    ierr = MatMult(mat,y,jac->work1);CHKERRQ(ierr);
    ierr = VecWAXPY(jac->work2,-1.0,jac->work1,x);CHKERRQ(ierr);
    ierr = VecSet(jac->work1,0.0);CHKERRQ(ierr);  /* zero since some PC's may not set all entries in the result */        
    ierr = PCApply(next->pc,jac->work2,jac->work1);CHKERRQ(ierr);
    ierr = VecAXPY(y,1.0,jac->work1);CHKERRQ(ierr);
  }
  if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    while (next->previous) {
      next = next->previous;
      ierr  = MatMult(mat,y,jac->work1);CHKERRQ(ierr);
      ierr = VecWAXPY(jac->work2,-1.0,jac->work1,x);CHKERRQ(ierr);
      ierr = VecSet(jac->work1,0.0);CHKERRQ(ierr);  /* zero since some PC's may not set all entries in the result */        
      ierr = PCApply(next->pc,jac->work2,jac->work1);CHKERRQ(ierr);
      ierr = VecAXPY(y,1.0,jac->work1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

/*
    This is very special for a matrix of the form alpha I + R + S
where first preconditioner is built from alpha I + S and second from
alpha I + R
*/
#undef __FUNCT__  
#define __FUNCT__ "PCApply_Composite_Special"
static PetscErrorCode PCApply_Composite_Special(PC pc,Vec x,Vec y)
{
  PetscErrorCode   ierr;
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  if (!next) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPC() or -pc_composite_pcs");
  }
  if (!next->next || next->next->next) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"Special composite preconditioners requires exactly two PCs");
  }

  ierr = PCApply(next->pc,x,jac->work1);CHKERRQ(ierr);
  ierr = PCApply(next->next->pc,jac->work1,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCApply_Composite_Additive"
static PetscErrorCode PCApply_Composite_Additive(PC pc,Vec x,Vec y)
{
  PetscErrorCode   ierr;
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  if (!next) {
    SETERRQ(PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPC() or -pc_composite_pcs");
  }
  ierr = PCApply(next->pc,x,y);CHKERRQ(ierr);
  while (next->next) {
    next = next->next;
    ierr = VecSet(jac->work1,0.0);CHKERRQ(ierr);  /* zero since some PC's may not set all entries in the result */        
    ierr = PCApply(next->pc,x,jac->work1);CHKERRQ(ierr);
    ierr = VecAXPY(y,1.0,jac->work1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetUp_Composite"
static PetscErrorCode PCSetUp_Composite(PC pc)
{
  PetscErrorCode   ierr;
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  if (!jac->work1) {
   ierr = MatGetVecs(pc->pmat,&jac->work1,0);CHKERRQ(ierr);
  }
  while (next) {
    ierr = PCSetOperators(next->pc,pc->mat,pc->pmat,pc->flag);CHKERRQ(ierr);
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCDestroy_Composite"
static PetscErrorCode PCDestroy_Composite(PC pc)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PetscErrorCode   ierr;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  while (next) {
    ierr = PCDestroy(next->pc);CHKERRQ(ierr);
    next = next->next;
  }

  if (jac->work1) {ierr = VecDestroy(jac->work1);CHKERRQ(ierr);}
  if (jac->work2) {ierr = VecDestroy(jac->work2);CHKERRQ(ierr);}
  ierr = PetscFree(jac);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCSetFromOptions_Composite"
static PetscErrorCode PCSetFromOptions_Composite(PC pc)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PetscErrorCode   ierr;
  PetscInt         nmax = 8,i;
  PC_CompositeLink next;
  char             *pcs[8];
  PetscTruth       flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead("Composite preconditioner options");CHKERRQ(ierr);
    ierr = PetscOptionsEnum("-pc_composite_type","Type of composition","PCCompositeSetType",PCCompositeTypes,(PetscEnum)jac->type,(PetscEnum*)&jac->type,&flg);CHKERRQ(ierr);
    if (flg) {
      ierr = PCCompositeSetType(pc,jac->type);CHKERRQ(ierr);
    }
    flg  = PETSC_FALSE;
    ierr = PetscOptionsTruth("-pc_composite_true","Use true matrix for inner solves","PCCompositeSetUseTrue",flg,&flg,PETSC_NULL);CHKERRQ(ierr);
    if (flg) {
      ierr = PCCompositeSetUseTrue(pc);CHKERRQ(ierr);
    }
    ierr = PetscOptionsStringArray("-pc_composite_pcs","List of composite solvers","PCCompositeAddPC",pcs,&nmax,&flg);CHKERRQ(ierr);
    if (flg) {
      for (i=0; i<nmax; i++) {
        ierr = PCCompositeAddPC(pc,pcs[i]);CHKERRQ(ierr);
      }
    }
  ierr = PetscOptionsTail();CHKERRQ(ierr);

  next = jac->head;
  while (next) {
    ierr = PCSetFromOptions(next->pc);CHKERRQ(ierr);
    next = next->next;
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCView_Composite"
static PetscErrorCode PCView_Composite(PC pc,PetscViewer viewer)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PetscErrorCode   ierr;
  PC_CompositeLink next = jac->head;
  PetscTruth       iascii;

  PetscFunctionBegin;
  ierr = PetscTypeCompare((PetscObject)viewer,PETSC_VIEWER_ASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Composite PC type - %s\n",PCCompositeTypes[jac->type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"PCs on composite preconditioner follow\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"---------------------------------\n");CHKERRQ(ierr);
  } else {
    SETERRQ1(PETSC_ERR_SUP,"Viewer type %s not supported for PCComposite",((PetscObject)viewer)->type_name);
  }
  if (iascii) {
    ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  }
  while (next) {
    ierr = PCView(next->pc,viewer);CHKERRQ(ierr);
    next = next->next;
  }
  if (iascii) {
    ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"---------------------------------\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCompositeSpecialSetAlpha_Composite"
PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeSpecialSetAlpha_Composite(PC pc,PetscScalar alpha)
{
  PC_Composite *jac = (PC_Composite*)pc->data;
  PetscFunctionBegin;
  jac->alpha = alpha;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCompositeSetType_Composite"
PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeSetType_Composite(PC pc,PCCompositeType type)
{
  PetscFunctionBegin;
  if (type == PC_COMPOSITE_ADDITIVE) {
    pc->ops->apply = PCApply_Composite_Additive;
  } else if (type ==  PC_COMPOSITE_MULTIPLICATIVE || type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    pc->ops->apply = PCApply_Composite_Multiplicative;
  } else if (type ==  PC_COMPOSITE_SPECIAL) {
    pc->ops->apply = PCApply_Composite_Special;
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Unkown composite preconditioner type");
  }
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCompositeAddPC_Composite"
PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeAddPC_Composite(PC pc,PCType type)
{
  PC_Composite     *jac;
  PC_CompositeLink next,ilink;
  PetscErrorCode   ierr;
  PetscInt         cnt = 0;
  const char       *prefix;
  char             newprefix[8];

  PetscFunctionBegin;
  ierr        = PetscNewLog(pc,struct _PC_CompositeLink,&ilink);CHKERRQ(ierr);
  ilink->next = 0;
  ierr = PCCreate(((PetscObject)pc)->comm,&ilink->pc);CHKERRQ(ierr);

  jac  = (PC_Composite*)pc->data;
  next = jac->head;
  if (!next) {
    jac->head       = ilink;
    ilink->previous = PETSC_NULL;
  } else {
    cnt++;
    while (next->next) {
      next = next->next;
      cnt++;
    }
    next->next      = ilink;
    ilink->previous = next;
  }
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(ilink->pc,prefix);CHKERRQ(ierr);
  sprintf(newprefix,"sub_%d_",(int)cnt);
  ierr = PCAppendOptionsPrefix(ilink->pc,newprefix);CHKERRQ(ierr);
  /* type is set after prefix, because some methods may modify prefix, e.g. pcksp */
  ierr = PCSetType(ilink->pc,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCompositeGetPC_Composite"
PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeGetPC_Composite(PC pc,PetscInt n,PC *subpc)
{
  PC_Composite     *jac;
  PC_CompositeLink next;
  PetscInt         i;

  PetscFunctionBegin;
  jac  = (PC_Composite*)pc->data;
  next = jac->head;
  for (i=0; i<n; i++) {
    if (!next->next) {
      SETERRQ(PETSC_ERR_ARG_INCOMP,"Not enough PCs in composite preconditioner");
    }
    next = next->next;
  }
  *subpc = next->pc;
  PetscFunctionReturn(0);
}
EXTERN_C_END

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCompositeSetUseTrue_Composite"
PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeSetUseTrue_Composite(PC pc)
{
  PC_Composite   *jac;

  PetscFunctionBegin;
  jac                  = (PC_Composite*)pc->data;
  jac->use_true_matrix = PETSC_TRUE;
  PetscFunctionReturn(0);
}
EXTERN_C_END

/* -------------------------------------------------------------------------------- */
#undef __FUNCT__  
#define __FUNCT__ "PCCompositeSetType"
/*@
   PCCompositeSetType - Sets the type of composite preconditioner.
   
   Collective on PC

   Input Parameter:
+  pc - the preconditioner context
-  type - PC_COMPOSITE_ADDITIVE (default), PC_COMPOSITE_MULTIPLICATIVE, PC_COMPOSITE_SPECIAL

   Options Database Key:
.  -pc_composite_type <type: one of multiplicative, additive, special> - Sets composite preconditioner type

   Level: Developer

.keywords: PC, set, type, composite preconditioner, additive, multiplicative
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeSetType(PC pc,PCCompositeType type)
{
  PetscErrorCode ierr,(*f)(PC,PCCompositeType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCompositeSetType_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,type);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCCompositeSpecialSetAlpha"
/*@
   PCCompositeSpecialSetAlpha - Sets alpha for the special composite preconditioner
     for alphaI + R + S
   
   Collective on PC

   Input Parameter:
+  pc - the preconditioner context
-  alpha - scale on identity

   Level: Developer

.keywords: PC, set, type, composite preconditioner, additive, multiplicative
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeSpecialSetAlpha(PC pc,PetscScalar alpha)
{
  PetscErrorCode ierr,(*f)(PC,PetscScalar);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCompositeSpecialSetAlpha_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,alpha);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCCompositeAddPC"
/*@C
   PCCompositeAddPC - Adds another PC to the composite PC.
   
   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - the type of the new preconditioner

   Level: Developer

.keywords: PC, composite preconditioner, add
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeAddPC(PC pc,PCType type)
{
  PetscErrorCode ierr,(*f)(PC,PCType);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCompositeAddPC_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,type);CHKERRQ(ierr);
  } 
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCCompositeGetPC"
/*@
   PCCompositeGetPC - Gets one of the PC objects in the composite PC.
   
   Not Collective

   Input Parameter:
+  pc - the preconditioner context
-  n - the number of the pc requested

   Output Parameters:
.  subpc - the PC requested

   Level: Developer

.keywords: PC, get, composite preconditioner, sub preconditioner

.seealso: PCCompositeAddPC()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeGetPC(PC pc,PetscInt n,PC *subpc)
{
  PetscErrorCode ierr,(*f)(PC,PetscInt,PC *);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  PetscValidPointer(subpc,3);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCompositeGetPC_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc,n,subpc);CHKERRQ(ierr);
  } else {
    SETERRQ(PETSC_ERR_ARG_WRONG,"Cannot get pc, not composite type");
  }
  PetscFunctionReturn(0);
}

#undef __FUNCT__  
#define __FUNCT__ "PCCompositeSetUseTrue"
/*@
   PCCompositeSetUseTrue - Sets a flag to indicate that the true matrix (rather than
                      the matrix used to define the preconditioner) is used to compute
                      the residual when the multiplicative scheme is used.

   Collective on PC

   Input Parameters:
.  pc - the preconditioner context

   Options Database Key:
.  -pc_composite_true - Activates PCCompositeSetUseTrue()

   Note:
   For the common case in which the preconditioning and linear 
   system matrices are identical, this routine is unnecessary.

   Level: Developer

.keywords: PC, composite preconditioner, set, true, flag

.seealso: PCSetOperators(), PCBJacobiSetUseTrueLocal(), PCKSPSetUseTrue()
@*/
PetscErrorCode PETSCKSP_DLLEXPORT PCCompositeSetUseTrue(PC pc)
{
  PetscErrorCode ierr,(*f)(PC);

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_COOKIE,1);
  ierr = PetscObjectQueryFunction((PetscObject)pc,"PCCompositeSetUseTrue_C",(void (**)(void))&f);CHKERRQ(ierr);
  if (f) {
    ierr = (*f)(pc);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/

/*MC
     PCCOMPOSITE - Build a preconditioner by composing together several preconditioners 

   Options Database Keys:
+  -pc_composite_type <type: one of multiplicative, additive, special> - Sets composite preconditioner type
.  -pc_composite_true - Activates PCCompositeSetUseTrue()
-  -pc_composite_pcs - <pc0,pc1,...> list of PCs to compose

   Level: intermediate

   Concepts: composing solvers

   Notes: To use a Krylov method inside the composite preconditioner, set the PCType of one or more
          inner PCs to be PCKSP. 
          Using a Krylov method inside another Krylov method can be dangerous (you get divergence or
          the incorrect answer) unless you use KSPFGMRES as the outter Krylov method


.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCSHELL, PCKSP, PCCompositeSetType(), PCCompositeSpecialSetAlpha(), PCCompositeAddPC(),
           PCCompositeGetPC(), PCCompositeSetUseTrue()

M*/

EXTERN_C_BEGIN
#undef __FUNCT__  
#define __FUNCT__ "PCCreate_Composite"
PetscErrorCode PETSCKSP_DLLEXPORT PCCreate_Composite(PC pc)
{
  PetscErrorCode ierr;
  PC_Composite   *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,PC_Composite,&jac);CHKERRQ(ierr);
  pc->ops->apply              = PCApply_Composite_Additive;
  pc->ops->setup              = PCSetUp_Composite;
  pc->ops->destroy            = PCDestroy_Composite;
  pc->ops->setfromoptions     = PCSetFromOptions_Composite;
  pc->ops->view               = PCView_Composite;
  pc->ops->applyrichardson    = 0;

  pc->data               = (void*)jac;
  jac->type              = PC_COMPOSITE_ADDITIVE;
  jac->work1             = 0;
  jac->work2             = 0;
  jac->head              = 0;

  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCompositeSetType_C","PCCompositeSetType_Composite",
                    PCCompositeSetType_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCompositeAddPC_C","PCCompositeAddPC_Composite",
                    PCCompositeAddPC_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCompositeGetPC_C","PCCompositeGetPC_Composite",
                    PCCompositeGetPC_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCompositeSetUseTrue_C","PCCompositeSetUseTrue_Composite",
                    PCCompositeSetUseTrue_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunctionDynamic((PetscObject)pc,"PCCompositeSpecialSetAlpha_C","PCCompositeSpecialSetAlpha_Composite",
                    PCCompositeSpecialSetAlpha_Composite);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}
EXTERN_C_END

