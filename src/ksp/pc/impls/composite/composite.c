
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
  PetscErrorCode   ierr;
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;
  Mat              mat  = pc->pmat;

  PetscFunctionBegin;

  if (!next) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPCType() or -pc_composite_pcs");

  /* Set the reuse flag on children PCs */
  while (next) {
    ierr = PCSetReusePreconditioner(next->pc,pc->reusepreconditioner);CHKERRQ(ierr);
    next = next->next;
  }
  next = jac->head;

  if (next->next && !jac->work2) { /* allocate second work vector */
    ierr = VecDuplicate(jac->work1,&jac->work2);CHKERRQ(ierr);
  }
  if (pc->useAmat) mat = pc->mat;
  ierr = PCApply(next->pc,x,y);CHKERRQ(ierr);                      /* y <- B x */
  while (next->next) {
    next = next->next;
    ierr = MatMult(mat,y,jac->work1);CHKERRQ(ierr);                /* work1 <- A y */
    ierr = VecWAXPY(jac->work2,-1.0,jac->work1,x);CHKERRQ(ierr);   /* work2 <- x - work1 */
    ierr = PCApply(next->pc,jac->work2,jac->work1);CHKERRQ(ierr);  /* work1 <- C work2 */
    ierr = VecAXPY(y,1.0,jac->work1);CHKERRQ(ierr);                /* y <- y + work1 = B x + C (x - A B x) = (B + C (1 - A B)) x */
  }
  if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    while (next->previous) {
      next = next->previous;
      ierr = MatMult(mat,y,jac->work1);CHKERRQ(ierr);
      ierr = VecWAXPY(jac->work2,-1.0,jac->work1,x);CHKERRQ(ierr);
      ierr = PCApply(next->pc,jac->work2,jac->work1);CHKERRQ(ierr);
      ierr = VecAXPY(y,1.0,jac->work1);CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_Composite_Multiplicative(PC pc,Vec x,Vec y)
{
  PetscErrorCode   ierr;
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;
  Mat              mat  = pc->pmat;

  PetscFunctionBegin;
  if (!next) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPCType() or -pc_composite_pcs");
  if (next->next && !jac->work2) { /* allocate second work vector */
    ierr = VecDuplicate(jac->work1,&jac->work2);CHKERRQ(ierr);
  }
  if (pc->useAmat) mat = pc->mat;
  /* locate last PC */
  while (next->next) {
    next = next->next;
  }
  ierr = PCApplyTranspose(next->pc,x,y);CHKERRQ(ierr);
  while (next->previous) {
    next = next->previous;
    ierr = MatMultTranspose(mat,y,jac->work1);CHKERRQ(ierr);
    ierr = VecWAXPY(jac->work2,-1.0,jac->work1,x);CHKERRQ(ierr);
    ierr = PCApplyTranspose(next->pc,jac->work2,jac->work1);CHKERRQ(ierr);
    ierr = VecAXPY(y,1.0,jac->work1);CHKERRQ(ierr);
  }
  if (jac->type == PC_COMPOSITE_SYMMETRIC_MULTIPLICATIVE) {
    next = jac->head;
    while (next->next) {
      next = next->next;
      ierr = MatMultTranspose(mat,y,jac->work1);CHKERRQ(ierr);
      ierr = VecWAXPY(jac->work2,-1.0,jac->work1,x);CHKERRQ(ierr);
      ierr = PCApplyTranspose(next->pc,jac->work2,jac->work1);CHKERRQ(ierr);
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
static PetscErrorCode PCApply_Composite_Special(PC pc,Vec x,Vec y)
{
  PetscErrorCode   ierr;
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  if (!next) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPCType() or -pc_composite_pcs");
  if (!next->next || next->next->next) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"Special composite preconditioners requires exactly two PCs");

  /* Set the reuse flag on children PCs */
  ierr = PCSetReusePreconditioner(next->pc,pc->reusepreconditioner);CHKERRQ(ierr);
  ierr = PCSetReusePreconditioner(next->next->pc,pc->reusepreconditioner);CHKERRQ(ierr);

  ierr = PCApply(next->pc,x,jac->work1);CHKERRQ(ierr);
  ierr = PCApply(next->next->pc,jac->work1,y);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_Composite_Additive(PC pc,Vec x,Vec y)
{
  PetscErrorCode   ierr;
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  if (!next) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPCType() or -pc_composite_pcs");

  /* Set the reuse flag on children PCs */
  while (next) {
    ierr = PCSetReusePreconditioner(next->pc,pc->reusepreconditioner);CHKERRQ(ierr);
    next = next->next;
  }
  next = jac->head;

  ierr = PCApply(next->pc,x,y);CHKERRQ(ierr);
  while (next->next) {
    next = next->next;
    ierr = PCApply(next->pc,x,jac->work1);CHKERRQ(ierr);
    ierr = VecAXPY(y,1.0,jac->work1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_Composite_Additive(PC pc,Vec x,Vec y)
{
  PetscErrorCode   ierr;
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  if (!next) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONGSTATE,"No composite preconditioners supplied via PCCompositeAddPCType() or -pc_composite_pcs");
  ierr = PCApplyTranspose(next->pc,x,y);CHKERRQ(ierr);
  while (next->next) {
    next = next->next;
    ierr = PCApplyTranspose(next->pc,x,jac->work1);CHKERRQ(ierr);
    ierr = VecAXPY(y,1.0,jac->work1);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_Composite(PC pc)
{
  PetscErrorCode   ierr;
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PC_CompositeLink next = jac->head;
  DM               dm;

  PetscFunctionBegin;
  if (!jac->work1) {
    ierr = MatCreateVecs(pc->pmat,&jac->work1,NULL);CHKERRQ(ierr);
  }
  ierr = PCGetDM(pc,&dm);CHKERRQ(ierr);
  while (next) {
    if (!next->pc->dm) {
      ierr = PCSetDM(next->pc,dm);CHKERRQ(ierr);
    }
    if (!next->pc->mat) {
      ierr = PCSetOperators(next->pc,pc->mat,pc->pmat);CHKERRQ(ierr);
    }
    next = next->next;
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCReset_Composite(PC pc)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PetscErrorCode   ierr;
  PC_CompositeLink next = jac->head;

  PetscFunctionBegin;
  while (next) {
    ierr = PCReset(next->pc);CHKERRQ(ierr);
    next = next->next;
  }
  ierr = VecDestroy(&jac->work1);CHKERRQ(ierr);
  ierr = VecDestroy(&jac->work2);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_Composite(PC pc)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PetscErrorCode   ierr;
  PC_CompositeLink next = jac->head,next_tmp;

  PetscFunctionBegin;
  ierr = PCReset_Composite(pc);CHKERRQ(ierr);
  while (next) {
    ierr     = PCDestroy(&next->pc);CHKERRQ(ierr);
    next_tmp = next;
    next     = next->next;
    ierr     = PetscFree(next_tmp);CHKERRQ(ierr);
  }
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_Composite(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PetscErrorCode   ierr;
  PetscInt         nmax = 8,i;
  PC_CompositeLink next;
  char             *pcs[8];
  PetscBool        flg;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"Composite preconditioner options");CHKERRQ(ierr);
  ierr = PetscOptionsEnum("-pc_composite_type","Type of composition","PCCompositeSetType",PCCompositeTypes,(PetscEnum)jac->type,(PetscEnum*)&jac->type,&flg);CHKERRQ(ierr);
  if (flg) {
    ierr = PCCompositeSetType(pc,jac->type);CHKERRQ(ierr);
  }
  ierr = PetscOptionsStringArray("-pc_composite_pcs","List of composite solvers","PCCompositeAddPCType",pcs,&nmax,&flg);CHKERRQ(ierr);
  if (flg) {
    for (i=0; i<nmax; i++) {
      ierr = PCCompositeAddPCType(pc,pcs[i]);CHKERRQ(ierr);
      ierr = PetscFree(pcs[i]);CHKERRQ(ierr);   /* deallocate string pcs[i], which is allocated in PetscOptionsStringArray() */
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

static PetscErrorCode PCView_Composite(PC pc,PetscViewer viewer)
{
  PC_Composite     *jac = (PC_Composite*)pc->data;
  PetscErrorCode   ierr;
  PC_CompositeLink next = jac->head;
  PetscBool        iascii;

  PetscFunctionBegin;
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"Composite PC type - %s\n",PCCompositeTypes[jac->type]);CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"PCs on composite preconditioner follow\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"---------------------------------\n");CHKERRQ(ierr);
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
  PetscErrorCode   ierr;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc, &ilink);CHKERRQ(ierr);
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
  ierr = PCGetOptionsPrefix(pc, &prefix);CHKERRQ(ierr);
  ierr = PCSetOptionsPrefix(subpc, prefix);CHKERRQ(ierr);
  ierr = PetscSNPrintf(newprefix, 20, "sub_%d_", (int) cnt);CHKERRQ(ierr);
  ierr = PCAppendOptionsPrefix(subpc, newprefix);CHKERRQ(ierr);
  ierr = PetscObjectReference((PetscObject) subpc);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCCompositeAddPCType_Composite(PC pc, PCType type)
{
  PC             subpc;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PCCreate(PetscObjectComm((PetscObject)pc), &subpc);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject) subpc, (PetscObject) pc, 1);CHKERRQ(ierr);
  ierr = PetscLogObjectParent((PetscObject) pc, (PetscObject) subpc);CHKERRQ(ierr);
  ierr = PCCompositeAddPC_Composite(pc, subpc);CHKERRQ(ierr);
  /* type is set after prefix, because some methods may modify prefix, e.g. pcksp */
  ierr = PCSetType(subpc, type);CHKERRQ(ierr);
  ierr = PCDestroy(&subpc);CHKERRQ(ierr);
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
    if (!next->next) SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_INCOMP,"Not enough PCs in composite preconditioner");
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveEnum(pc,type,2);
  ierr = PetscTryMethod(pc,"PCCompositeSetType_C",(PC,PCCompositeType),(pc,type));CHKERRQ(ierr);
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
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscUseMethod(pc,"PCCompositeGetType_C",(PC,PCCompositeType*),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCCompositeSpecialSetAlpha - Sets alpha for the special composite preconditioner
     for alphaI + R + S

   Logically Collective on PC

   Input Parameter:
+  pc - the preconditioner context
-  alpha - scale on identity

   Level: Developer

@*/
PetscErrorCode  PCCompositeSpecialSetAlpha(PC pc,PetscScalar alpha)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidLogicalCollectiveScalar(pc,alpha,2);
  ierr = PetscTryMethod(pc,"PCCompositeSpecialSetAlpha_C",(PC,PetscScalar),(pc,alpha));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@C
  PCCompositeAddPCType - Adds another PC of the given type to the composite PC.

  Collective on PC

  Input Parameters:
+ pc - the preconditioner context
- type - the type of the new preconditioner

  Level: Developer

.seealso: PCCompositeAddPC()
@*/
PetscErrorCode  PCCompositeAddPCType(PC pc,PCType type)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PetscTryMethod(pc,"PCCompositeAddPCType_C",(PC,PCType),(pc,type));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
  PCCompositeAddPC - Adds another PC to the composite PC.

  Collective on PC

  Input Parameters:
+ pc    - the preconditioner context
- subpc - the new preconditioner

   Level: Developer

.seealso: PCCompositeAddPCType()
@*/
PetscErrorCode PCCompositeAddPC(PC pc, PC subpc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(subpc,PC_CLASSID,2);
  ierr = PetscTryMethod(pc,"PCCompositeAddPC_C",(PC,PC),(pc,subpc));CHKERRQ(ierr);
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

.seealso: PCCompositeGetPC()
@*/
PetscErrorCode  PCCompositeGetNumberPC(PC pc,PetscInt *num)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(num,2);
  ierr = PetscUseMethod(pc,"PCCompositeGetNumberPC_C",(PC,PetscInt*),(pc,num));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/*@
   PCCompositeGetPC - Gets one of the PC objects in the composite PC.

   Not Collective

   Input Parameter:
+  pc - the preconditioner context
-  n - the number of the pc requested

   Output Parameters:
.  subpc - the PC requested

   Level: Developer

    Notes:
    To use a different operator to construct one of the inner preconditioners first call PCCompositeGetPC(), then
            call PCSetOperators() on that PC.

.seealso: PCCompositeAddPCType(), PCCompositeGetNumberPC(), PCSetOperators()
@*/
PetscErrorCode PCCompositeGetPC(PC pc,PetscInt n,PC *subpc)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(subpc,3);
  ierr = PetscUseMethod(pc,"PCCompositeGetPC_C",(PC,PetscInt,PC*),(pc,n,subpc));CHKERRQ(ierr);
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

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCSHELL, PCKSP, PCCompositeSetType(), PCCompositeSpecialSetAlpha(), PCCompositeAddPCType(),
           PCCompositeGetPC(), PCSetUseAmat()

M*/

PETSC_EXTERN PetscErrorCode PCCreate_Composite(PC pc)
{
  PetscErrorCode ierr;
  PC_Composite   *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&jac);CHKERRQ(ierr);

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

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeSetType_C",PCCompositeSetType_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeGetType_C",PCCompositeGetType_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeAddPCType_C",PCCompositeAddPCType_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeAddPC_C",PCCompositeAddPC_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeGetNumberPC_C",PCCompositeGetNumberPC_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeGetPC_C",PCCompositeGetPC_Composite);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCCompositeSpecialSetAlpha_C",PCCompositeSpecialSetAlpha_Composite);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
