
/*
      Defines a preconditioner that can consist of a collection of PCs
*/
#include <petsc/private/pcimpl.h>
#include <petscksp.h>            /*I "petscksp.h" I*/

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
} PC_Composite;

static PetscErrorCode PCApply_Composite_Multiplicative(PC pc,Vec x,Vec y)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;
  Mat              mat  = pc->pmat;

  PetscFunctionBegin;

  PetscCheck(next,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPCType() or -pc_composite_pcs");

  /* Set the reuse flag on children PCs */
  while (next) {
    PetscCall(PCSetReusePreconditioner(next->pc,pc->reusepreconditioner));
    next = next->next;
  }
  next = jac->head;

  if (next->next && !jac->work2) { /* allocate second work vector */
    PetscCall(VecDuplicate(jac->work1,&jac->work2));
  }
  if (pc->useAmat) mat = pc->mat;
  PetscCall(PCApply(next->pc,x,y));                      /* y <- B x */
  while (next->next) {
    next = next->next;
    PetscCall(MatMult(mat,y,jac->work1));                /* work1 <- A y */
    PetscCall(VecWAXPY(jac->work2,-1.0,jac->work1,x));   /* work2 <- x - work1 */
    PetscCall(PCApply(next->pc,jac->work2,jac->work1));  /* work1 <- C work2 */
    PetscCall(VecAXPY(y,1.0,jac->work1));                /* y <- y + work1 = B x + C (x - A B x) = (B + C (1 - A B)) x */
  }
  if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    while (next->previous) {
      next = next->previous;
      PetscCall(MatMult(mat,y,jac->work1));
      PetscCall(VecWAXPY(jac->work2,-1.0,jac->work1,x));
      PetscCall(PCApply(next->pc,jac->work2,jac->work1));
      PetscCall(VecAXPY(y,1.0,jac->work1));
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_Composite_Multiplicative(PC pc,Vec x,Vec y)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;
  Mat              mat  = pc->pmat;

  PetscFunctionBegin;
  PetscCheck(next,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPCType() or -pc_composite_pcs");
  if (next->next && !jac->work2) { /* allocate second work vector */
    PetscCall(VecDuplicate(jac->work1,&jac->work2));
  }
  if (pc->useAmat) mat = pc->mat;
  /* locate last PC */
  while (next->next) {
    next = next->next;
  }
  PetscCall(PCApplyTranspose(next->pc,x,y));
  while (next->previous) {
    next = next->previous;
    PetscCall(MatMultTranspose(mat,y,jac->work1));
    PetscCall(VecWAXPY(jac->work2,-1.0,jac->work1,x));
    PetscCall(PCApplyTranspose(next->pc,jac->work2,jac->work1));
    PetscCall(VecAXPY(y,1.0,jac->work1));
  }
  if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    next = jac->head;
    while (next->next) {
      next = next->next;
      PetscCall(MatMultTranspose(mat,y,jac->work1));
      PetscCall(VecWAXPY(jac->work2,-1.0,jac->work1,x));
      PetscCall(PCApplyTranspose(next->pc,jac->work2,jac->work1));
      PetscCall(VecAXPY(y,1.0,jac->work1));
    }
  }
  PetscFunctionReturn(0);
}

/*
    This is very special for a matrix of the form alpha I + R + S
where first preconditioner is built from alpha I + S and second from
alpha I + R
*/
static PetscErrorCode PCApply_Composite_Special(PC pc,Vec x,Vec y)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  PetscCheck(next,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPCType() or -pc_composite_pcs");
  PetscCheck(next->next && !next->next->next,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Special composite preconditioners requires exactly two PCs");

  /* Set the reuse flag on children PCs */
  PetscCall(PCSetReusePreconditioner(next->pc,pc->reusepreconditioner));
  PetscCall(PCSetReusePreconditioner(next->next->pc,pc->reusepreconditioner));

  PetscCall(PCApply(next->pc,x,jac->work1));
  PetscCall(PCApply(next->next->pc,jac->work1,y));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Composite_Additive(PC pc,Vec x,Vec y)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  PetscCheck(next,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPCType() or -pc_composite_pcs");

  /* Set the reuse flag on children PCs */
  while (next) {
    PetscCall(PCSetReusePreconditioner(next->pc,pc->reusepreconditioner));
    next = next->next;
  }
  next = jac->head;

  PetscCall(PCApply(next->pc,x,y));
  while (next->next) {
    next = next->next;
    PetscCall(PCApply(next->pc,x,jac->work1));
    PetscCall(VecAXPY(y,1.0,jac->work1));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_Composite_Additive(PC pc,Vec x,Vec y)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  PetscCheck(next,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPCType() or -pc_composite_pcs");
  PetscCall(PCApplyTranspose(next->pc,x,y));
  while (next->next) {
    next = next->next;
    PetscCall(PCApplyTranspose(next->pc,x,jac->work1));
    PetscCall(VecAXPY(y,1.0,jac->work1));
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Composite(PC pc)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;
  DM               dm;

  PetscFunctionBegin;
  if (!jac->work1) {
    PetscCall(MatCreateVecs(pc->pmat,&jac->work1,NULL));
  }
  PetscCall(PCGetDM(pc,&dm));
  while (next) {
    if (!next->pc->dm) {
      PetscCall(PCSetDM(next->pc,dm));
    }
    if (!next->pc->mat) {
      PetscCall(PCSetOperators(next->pc,pc->mat,pc->pmat));
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Composite(PC pc)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  while (next) {
    PetscCall(PCReset(next->pc));
    next = next->next;
  }
  PetscCall(VecDestroy(&jac->work1));
  PetscCall(VecDestroy(&jac->work2));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Composite(PC pc)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head,next_tmp;

  PetscFunctionBegin;
  PetscCall(PCReset_Composite(pc));
  while (next) {
    PetscCall(PCDestroy(&next->pc));
    next_tmp = next;
    next     = next->next;
    PetscCall(PetscFree(next_tmp));
  }
  PetscCall(PetscFree(pc->data));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Composite(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PetscInt         nmax = 8,i;
  PC_CompositeLink next;
  char             *pcs[8];
  PetscBool        flg;

  PetscFunctionBegin;
  PetscOptionsHeadBegin(PetscOptionsObject,"Composite preconditioner options");
  PetscCall(PetscOptionsEnum("-pc_composite_type","Type of composition","PCCompositeSetType",PCCompositeTypes,(PetscEnum)jac->type,(PetscEnum*)&jac->type,&flg));
  if (flg) {
    PetscCall(PCCompositeSetType(pc,jac->type));
  }
  PetscCall(PetscOptionsStringArray("-pc_composite_pcs","List of composite solvers","PCCompositeAddPCType",pcs,&nmax,&flg));
  if (flg) {
    for (i=0; i<nmax; i++) {
      PetscCall(PCCompositeAddPCType(pc,pcs[i]));
      PetscCall(PetscFree(pcs[i]));   /* deallocate string pcs[i], which is allocated in PetscOptionsStringArray() */
    }
  }
  PetscOptionsHeadEnd();

  next = jac->head;
  while (next) {
    PetscCall(PCSetFromOptions(next->pc));
    next = next->next;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_Composite(PC pc,PetscViewer viewer)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;
  PetscBool        iascii;

  PetscFunctionBegin;
  PetscCall(PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii));
  if (iascii) {
    PetscCall(PetscViewerASCIIPrintf(viewer,"Composite PC type - %s\n",PCCompositeTypes[jac->type]));
    PetscCall(PetscViewerASCIIPrintf(viewer,"PCs on composite preconditioner follow\n"));
    PetscCall(PetscViewerASCIIPrintf(viewer,"---------------------------------\n"));
  }
  if (iascii) {
    PetscCall(PetscViewerASCIIPushTab(viewer));
  }
  while (next) {
    PetscCall(PCView(next->pc,viewer));
    next = next->next;
  }
  if (iascii) {
    PetscCall(PetscViewerASCIIPopTab(viewer));
    PetscCall(PetscViewerASCIIPrintf(viewer,"---------------------------------\n"));
  }
  PetscFunctionReturn(0);
}

/* ------------------------------------------------------------------------------*/

static PetscErrorCode  PCCompositeSpecialSetAlpha_Composite(PC pc,PetscScalar alpha)
{
  PC_Composite *jac = (PC_Composite*)pc->data;

  PetscFunctionBegin;
  jac->alpha = alpha;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCCompositeSetType_Composite(PC pc,PCCompositeType type)
{
  PC_Composite *jac = (PC_Composite*)pc->data;

  PetscFunctionBegin;
  if (type == PC_COMPOSITE_ADDITIVE) {
    pc->ops->apply          = PCApply_Composite_Additive;
    pc->ops->applytranspose = PCApplyTranspose_Composite_Additive;
  } else if (type ==  PC_COMPOSITE_MULTIPLICATIVE || type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    pc->ops->apply          = PCApply_Composite_Multiplicative;
    pc->ops->applytranspose = PCApplyTranspose_Composite_Multiplicative;
  } else if (type ==  PC_COMPOSITE_SPECIAL) {
    pc->ops->apply          = PCApply_Composite_Special;
    pc->ops->applytranspose = NULL;
  } else SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"Unknown composite preconditioner type");
  jac->type = type;
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCCompositeGetType_Composite(PC pc,PCCompositeType *type)
{
  PC_Composite *jac = (PC_Composite*)pc->data;

  PetscFunctionBegin;
  *type = jac->type;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCCompositeAddPC_Composite(PC pc, PC subpc)
{
  PC_Composite    *jac;
  PC_CompositeLink next, ilink;
  PetscInt         cnt = 0;
  const char      *prefix;
  char             newprefix[20];

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc, &ilink));
  ilink->next = NULL;
  ilink->pc   = subpc;

  jac  = (PC_Composite *) pc->data;
  next = jac->head;
  if (!next) {
    jac->head       = ilink;
    ilink->previous = NULL;
  } else {
    cnt++;
    while (next->next) {
      next = next->next;
      cnt++;
    }
    next->next      = ilink;
    ilink->previous = next;
  }
  PetscCall(PCGetOptionsPrefix(pc, &prefix));
  PetscCall(PCSetOptionsPrefix(subpc, prefix));
  PetscCall(PetscSNPrintf(newprefix, 20, "sub_%d_", (int) cnt));
  PetscCall(PCAppendOptionsPrefix(subpc, newprefix));
  PetscCall(PetscObjectReference((PetscObject) subpc));
  PetscFunctionReturn(0);
}

static PetscErrorCode PCCompositeAddPCType_Composite(PC pc, PCType type)
{
  PC             subpc;

  PetscFunctionBegin;
  PetscCall(PCCreate(PetscObjectComm((PetscObject)pc), &subpc));
  PetscCall(PetscObjectIncrementTabLevel((PetscObject) subpc, (PetscObject) pc, 1));
  PetscCall(PetscLogObjectParent((PetscObject) pc, (PetscObject) subpc));
  PetscCall(PCCompositeAddPC_Composite(pc, subpc));
  /* type is set after prefix, because some methods may modify prefix, e.g. pcksp */
  PetscCall(PCSetType(subpc, type));
  PetscCall(PCDestroy(&subpc));
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCCompositeGetNumberPC_Composite(PC pc,PetscInt *n)
{
  PC_Composite     *jac;
  PC_CompositeLink next;

  PetscFunctionBegin;
  jac  = (PC_Composite*)pc->data;
  next = jac->head;
  *n = 0;
  while (next) {
    next = next->next;
    (*n) ++;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCCompositeGetPC_Composite(PC pc,PetscInt n,PC *subpc)
{
  PC_Composite     *jac;
  PC_CompositeLink next;
  PetscInt         i;

  PetscFunctionBegin;
  jac  = (PC_Composite*)pc->data;
  next = jac->head;
  for (i=0; i<n; i++) {
    PetscCheck(next->next,PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"Not enough PCs in composite preconditioner");
    next = next->next;
  }
  *subpc = next->pc;
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------- */
/*@
   PCCompositeSetType - Sets the type of composite preconditioner.

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  type - PC_COMPOSITE_ADDITIVE (default), PC_COMPOSITE_MULTIPLICATIVE, PC_COMPOSITE_SPECIAL

   Options Database Key:
.  -pc_composite_type <type: one of multiplicative, additive, special> - Sets composite preconditioner type

   Level: Developer

@*/
PetscErrorCode  PCCompositeSetType(PC pc,PCCompositeType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  PetscTryMethod(pc,"PCCompositeSetType_C",(PC,PCCompositeType),(pc,type));
  PetscFunctionReturn(0);
}

/*@
   PCCompositeGetType - Gets the type of composite preconditioner.

   Logically Collective on PC

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  type - PC_COMPOSITE_ADDITIVE (default), PC_COMPOSITE_MULTIPLICATIVE, PC_COMPOSITE_SPECIAL

   Options Database Key:
.  -pc_composite_type <type: one of multiplicative, additive, special> - Sets composite preconditioner type

   Level: Developer

@*/
PetscErrorCode  PCCompositeGetType(PC pc,PCCompositeType *type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscUseMethod(pc,"PCCompositeGetType_C",(PC,PCCompositeType*),(pc,type));
  PetscFunctionReturn(0);
}

/*@
   PCCompositeSpecialSetAlpha - Sets alpha for the special composite preconditioner
     for alphaI + R + S

   Logically Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  alpha - scale on identity

   Level: Developer

@*/
PetscErrorCode  PCCompositeSpecialSetAlpha(PC pc,PetscScalar alpha)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveScalar(pc,alpha,2);
  PetscTryMethod(pc,"PCCompositeSpecialSetAlpha_C",(PC,PetscScalar),(pc,alpha));
  PetscFunctionReturn(0);
}

/*@C
  PCCompositeAddPCType - Adds another PC of the given type to the composite PC.

  Collective on PC

  Input Parameters:
+ pc - the preconditioner context
- type - the type of the new preconditioner

  Level: Developer

.seealso: `PCCompositeAddPC()`
@*/
PetscErrorCode  PCCompositeAddPCType(PC pc,PCType type)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscTryMethod(pc,"PCCompositeAddPCType_C",(PC,PCType),(pc,type));
  PetscFunctionReturn(0);
}

/*@
  PCCompositeAddPC - Adds another PC to the composite PC.

  Collective on PC

  Input Parameters:
+ pc    - the preconditioner context
- subpc - the new preconditioner

   Level: Developer

.seealso: `PCCompositeAddPCType()`
@*/
PetscErrorCode PCCompositeAddPC(PC pc, PC subpc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(subpc,PC_CLASSID,2);
  PetscTryMethod(pc,"PCCompositeAddPC_C",(PC,PC),(pc,subpc));
  PetscFunctionReturn(0);
}

/*@
   PCCompositeGetNumberPC - Gets the number of PC objects in the composite PC.

   Not Collective

   Input Parameter:
.  pc - the preconditioner context

   Output Parameter:
.  num - the number of sub pcs

   Level: Developer

.seealso: `PCCompositeGetPC()`
@*/
PetscErrorCode  PCCompositeGetNumberPC(PC pc,PetscInt *num)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(num,2);
  PetscUseMethod(pc,"PCCompositeGetNumberPC_C",(PC,PetscInt*),(pc,num));
  PetscFunctionReturn(0);
}

/*@
   PCCompositeGetPC - Gets one of the PC objects in the composite PC.

   Not Collective

   Input Parameters:
+  pc - the preconditioner context
-  n - the number of the pc requested

   Output Parameters:
.  subpc - the PC requested

   Level: Developer

    Notes:
    To use a different operator to construct one of the inner preconditioners first call PCCompositeGetPC(), then
            call PCSetOperators() on that PC.

.seealso: `PCCompositeAddPCType()`, `PCCompositeGetNumberPC()`, `PCSetOperators()`
@*/
PetscErrorCode PCCompositeGetPC(PC pc,PetscInt n,PC *subpc)
{
  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(subpc,3);
  PetscUseMethod(pc,"PCCompositeGetPC_C",(PC,PetscInt,PC*),(pc,n,subpc));
  PetscFunctionReturn(0);
}

/* -------------------------------------------------------------------------------------------*/

/*MC
     PCCOMPOSITE - Build a preconditioner by composing together several preconditioners

   Options Database Keys:
+  -pc_composite_type <type: one of multiplicative, additive, symmetric_multiplicative, special> - Sets composite preconditioner type
.  -pc_use_amat - activates PCSetUseAmat()
-  -pc_composite_pcs - <pc0,pc1,...> list of PCs to compose

   Level: intermediate

   Notes:
    To use a Krylov method inside the composite preconditioner, set the PCType of one or more
          inner PCs to be PCKSP.
          Using a Krylov method inside another Krylov method can be dangerous (you get divergence or
          the incorrect answer) unless you use KSPFGMRES as the outer Krylov method
          To use a different operator to construct one of the inner preconditioners first call PCCompositeGetPC(), then
          call PCSetOperators() on that PC.

.seealso: `PCCreate()`, `PCSetType()`, `PCType`, `PC`,
          `PCSHELL`, `PCKSP`, `PCCompositeSetType()`, `PCCompositeSpecialSetAlpha()`, `PCCompositeAddPCType()`,
          `PCCompositeGetPC()`, `PCSetUseAmat()`

M*/

PETSC_EXTERN PetscErrorCode PCCreate_Composite(PC pc)
{
  PC_Composite   *jac;

  PetscFunctionBegin;
  PetscCall(PetscNewLog(pc,&jac));

  pc->ops->apply           = PCApply_Composite_Additive;
  pc->ops->applytranspose  = PCApplyTranspose_Composite_Additive;
  pc->ops->setup           = PCSetUp_Composite;
  pc->ops->reset           = PCReset_Composite;
  pc->ops->destroy         = PCDestroy_Composite;
  pc->ops->setfromoptions  = PCSetFromOptions_Composite;
  pc->ops->view            = PCView_Composite;
  pc->ops->applyrichardson = NULL;

  pc->data   = (void*)jac;
  jac->type  = PC_COMPOSITE_ADDITIVE;
  jac->work1 = NULL;
  jac->work2 = NULL;
  jac->head  = NULL;

  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCCompositeSetType_C",PCCompositeSetType_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCCompositeGetType_C",PCCompositeGetType_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCCompositeAddPCType_C",PCCompositeAddPCType_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCCompositeAddPC_C",PCCompositeAddPC_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCCompositeGetNumberPC_C",PCCompositeGetNumberPC_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCCompositeGetPC_C",PCCompositeGetPC_Composite));
  PetscCall(PetscObjectComposeFunction((PetscObject)pc,"PCCompositeSpecialSetAlpha_C",PCCompositeSpecialSetAlpha_Composite));
  PetscFunctionReturn(0);
}
