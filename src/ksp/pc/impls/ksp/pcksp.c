#include <petsc/private/pcimpl.h>
#include <petsc/private/kspimpl.h>
#include <petscksp.h>            /*I "petscksp.h" I*/

typedef struct {
  KSP      ksp;
  PetscInt its; /* total number of iterations KSP uses */
} PC_KSP;

static PetscErrorCode  PCKSPCreateKSP_KSP(PC pc)
{
  PetscErrorCode ierr;
  const char     *prefix;
  PC_KSP         *jac = (PC_KSP*)pc->data;
  DM             dm;

  PetscFunctionBegin;
  ierr = KSPCreate(PetscObjectComm((PetscObject)pc),&jac->ksp);CHKERRQ(ierr);
  ierr = KSPSetErrorIfNotConverged(jac->ksp,pc->erroriffailure);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)jac->ksp,(PetscObject)pc,1);CHKERRQ(ierr);
  ierr = PCGetOptionsPrefix(pc,&prefix);CHKERRQ(ierr);
  ierr = KSPSetOptionsPrefix(jac->ksp,prefix);CHKERRQ(ierr);
  ierr = KSPAppendOptionsPrefix(jac->ksp,"ksp_");CHKERRQ(ierr);
  ierr = PCGetDM(pc, &dm);CHKERRQ(ierr);
  if (dm) {
    ierr = KSPSetDM(jac->ksp, dm);CHKERRQ(ierr);
    ierr = KSPSetDMActive(jac->ksp, PETSC_FALSE);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApply_KSP(PC pc,Vec x,Vec y)
{
  PetscErrorCode     ierr;
  PetscInt           its;
  PC_KSP             *jac = (PC_KSP*)pc->data;

  PetscFunctionBegin;
  if (jac->ksp->presolve) {
    ierr = VecCopy(x,y);CHKERRQ(ierr);
    ierr = KSPSolve(jac->ksp,y,y);CHKERRQ(ierr);
  } else {
    ierr = KSPSolve(jac->ksp,x,y);CHKERRQ(ierr);
  }
  ierr = KSPCheckSolve(jac->ksp,pc,y);CHKERRQ(ierr);
  ierr      = KSPGetIterationNumber(jac->ksp,&its);CHKERRQ(ierr);
  jac->its += its;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCMatApply_KSP(PC pc,Mat X,Mat Y)
{
  PetscErrorCode     ierr;
  PetscInt           its;
  PC_KSP             *jac = (PC_KSP*)pc->data;

  PetscFunctionBegin;
  if (jac->ksp->presolve) {
    ierr = MatCopy(X,Y,SAME_NONZERO_PATTERN);CHKERRQ(ierr);
    ierr = KSPMatSolve(jac->ksp,Y,Y);CHKERRQ(ierr); /* TODO FIXME: this will fail since KSPMatSolve does not allow inplace solve yet */
  } else {
    ierr = KSPMatSolve(jac->ksp,X,Y);CHKERRQ(ierr);
  }
  ierr = KSPCheckSolve(jac->ksp,pc,NULL);CHKERRQ(ierr);
  ierr      = KSPGetIterationNumber(jac->ksp,&its);CHKERRQ(ierr);
  jac->its += its;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCApplyTranspose_KSP(PC pc,Vec x,Vec y)
{
  PetscErrorCode     ierr;
  PetscInt           its;
  PC_KSP             *jac = (PC_KSP*)pc->data;

  PetscFunctionBegin;
  if (jac->ksp->presolve) {
    ierr = VecCopy(x,y);CHKERRQ(ierr);
    ierr = KSPSolve(jac->ksp,y,y);CHKERRQ(ierr);
  } else {
    ierr = KSPSolveTranspose(jac->ksp,x,y);CHKERRQ(ierr);
  }
  ierr = KSPCheckSolve(jac->ksp,pc,y);CHKERRQ(ierr);
  ierr      = KSPGetIterationNumber(jac->ksp,&its);CHKERRQ(ierr);
  jac->its += its;
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetUp_KSP(PC pc)
{
  PetscErrorCode ierr;
  PC_KSP         *jac = (PC_KSP*)pc->data;
  Mat            mat;

  PetscFunctionBegin;
  if (!jac->ksp) {
    ierr = PCKSPCreateKSP_KSP(pc);CHKERRQ(ierr);
    ierr = KSPSetFromOptions(jac->ksp);CHKERRQ(ierr);
  }
  if (pc->useAmat) mat = pc->mat;
  else             mat = pc->pmat;
  ierr = KSPSetOperators(jac->ksp,mat,pc->pmat);CHKERRQ(ierr);
  ierr = KSPSetUp(jac->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* Default destroy, if it has never been setup */
static PetscErrorCode PCReset_KSP(PC pc)
{
  PC_KSP         *jac = (PC_KSP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy(&jac->ksp);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCDestroy_KSP(PC pc)
{
  PC_KSP         *jac = (PC_KSP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = KSPDestroy(&jac->ksp);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCKSPGetKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCKSPSetKSP_C",NULL);CHKERRQ(ierr);
  ierr = PetscFree(pc->data);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCView_KSP(PC pc,PetscViewer viewer)
{
  PC_KSP         *jac = (PC_KSP*)pc->data;
  PetscErrorCode ierr;
  PetscBool      iascii;

  PetscFunctionBegin;
  if (!jac->ksp) {ierr = PCKSPCreateKSP_KSP(pc);CHKERRQ(ierr);}
  ierr = PetscObjectTypeCompare((PetscObject)viewer,PETSCVIEWERASCII,&iascii);CHKERRQ(ierr);
  if (iascii) {
    if (pc->useAmat) {
      ierr = PetscViewerASCIIPrintf(viewer,"  Using Amat (not Pmat) as operator on inner solve\n");CHKERRQ(ierr);
    }
    ierr = PetscViewerASCIIPrintf(viewer,"  KSP and PC on KSP preconditioner follow\n");CHKERRQ(ierr);
    ierr = PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n");CHKERRQ(ierr);
  }
  ierr = PetscViewerASCIIPushTab(viewer);CHKERRQ(ierr);
  ierr = KSPView(jac->ksp,viewer);CHKERRQ(ierr);
  ierr = PetscViewerASCIIPopTab(viewer);CHKERRQ(ierr);
  if (iascii) {
    ierr = PetscViewerASCIIPrintf(viewer,"  ---------------------------------\n");CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCKSPSetKSP_KSP(PC pc,KSP ksp)
{
  PC_KSP         *jac = (PC_KSP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscObjectReference((PetscObject)ksp);CHKERRQ(ierr);
  ierr = KSPDestroy(&jac->ksp);CHKERRQ(ierr);
  jac->ksp = ksp;
  PetscFunctionReturn(0);
}

/*@
   PCKSPSetKSP - Sets the KSP context for a KSP PC.

   Collective on PC

   Input Parameters:
+  pc - the preconditioner context
-  ksp - the KSP solver

   Notes:
   The PC and the KSP must have the same communicator

   Level: advanced

@*/
PetscErrorCode  PCKSPSetKSP(PC pc,KSP ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(ksp,KSP_CLASSID,2);
  PetscCheckSameComm(pc,1,ksp,2);
  ierr = PetscTryMethod(pc,"PCKSPSetKSP_C",(PC,KSP),(pc,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode  PCKSPGetKSP_KSP(PC pc,KSP *ksp)
{
  PC_KSP         *jac = (PC_KSP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (!jac->ksp) {ierr = PCKSPCreateKSP_KSP(pc);CHKERRQ(ierr);}
  *ksp = jac->ksp;
  PetscFunctionReturn(0);
}

/*@
   PCKSPGetKSP - Gets the KSP context for a KSP PC.

   Not Collective but KSP returned is parallel if PC was parallel

   Input Parameter:
.  pc - the preconditioner context

   Output Parameters:
.  ksp - the KSP solver

   Notes:
   You must call KSPSetUp() before calling PCKSPGetKSP().

   If the PC is not a PCKSP object it raises an error

   Level: advanced

@*/
PetscErrorCode  PCKSPGetKSP(PC pc,KSP *ksp)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidPointer(ksp,2);
  ierr = PetscUseMethod(pc,"PCKSPGetKSP_C",(PC,KSP*),(pc,ksp));CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode PCSetFromOptions_KSP(PetscOptionItems *PetscOptionsObject,PC pc)
{
  PC_KSP         *jac = (PC_KSP*)pc->data;
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscOptionsHead(PetscOptionsObject,"PC KSP options");CHKERRQ(ierr);
  if (jac->ksp) {
    ierr = KSPSetFromOptions(jac->ksp);CHKERRQ(ierr);
   }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* ----------------------------------------------------------------------------------*/

/*MC
     PCKSP -    Defines a preconditioner that can consist of any KSP solver.
                 This allows, for example, embedding a Krylov method inside a preconditioner.

   Options Database Key:
.     -pc_use_amat - use the matrix that defines the linear system, Amat as the matrix for the
                    inner solver, otherwise by default it uses the matrix used to construct
                    the preconditioner, Pmat (see PCSetOperators())

   Level: intermediate

   Notes:
    The application of an inexact Krylov solve is a nonlinear operation. Thus, performing a solve with KSP is,
    in general, a nonlinear operation, so PCKSP is in general a nonlinear preconditioner.
    Thus, one can see divergence or an incorrect answer unless using a flexible Krylov method (e.g. KSPFGMRES, KSPGCR, or KSPFCG) for the outer Krylov solve.

   Developer Notes:
    If the outer Krylov method has a nonzero initial guess it will compute a new residual based on that initial guess
    and pass that as the right hand side into this KSP (and hence this KSP will always have a zero initial guess). For all outer Krylov methods
    except Richardson this is neccessary since Krylov methods, even the flexible ones, need to "see" the result of the action of the preconditioner on the
    input (current residual) vector, the action of the preconditioner cannot depend also on some other vector (the "initial guess"). For
    KSPRICHARDSON it is possible to provide a PCApplyRichardson_PCKSP() that short circuits returning to the KSP object at each iteration to compute the
    residual, see for example PCApplyRichardson_SOR(). We do not implement a PCApplyRichardson_PCKSP()  because (1) using a KSP directly inside a Richardson
    is not an efficient algorithm anyways and (2) implementing it for its > 1 would essentially require that we implement Richardson (reimplementing the
    Richardson code) inside the PCApplyRichardson_PCKSP() leading to duplicate code.

.seealso:  PCCreate(), PCSetType(), PCType (for list of available types), PC,
           PCSHELL, PCCOMPOSITE, PCSetUseAmat(), PCKSPGetKSP()

M*/

PETSC_EXTERN PetscErrorCode PCCreate_KSP(PC pc)
{
  PetscErrorCode ierr;
  PC_KSP         *jac;

  PetscFunctionBegin;
  ierr = PetscNewLog(pc,&jac);CHKERRQ(ierr);
  pc->data = (void*)jac;

  ierr = PetscMemzero(pc->ops,sizeof(struct _PCOps));CHKERRQ(ierr);
  pc->ops->apply           = PCApply_KSP;
  pc->ops->matapply        = PCMatApply_KSP;
  pc->ops->applytranspose  = PCApplyTranspose_KSP;
  pc->ops->setup           = PCSetUp_KSP;
  pc->ops->reset           = PCReset_KSP;
  pc->ops->destroy         = PCDestroy_KSP;
  pc->ops->setfromoptions  = PCSetFromOptions_KSP;
  pc->ops->view            = PCView_KSP;

  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCKSPGetKSP_C",PCKSPGetKSP_KSP);CHKERRQ(ierr);
  ierr = PetscObjectComposeFunction((PetscObject)pc,"PCKSPSetKSP_C",PCKSPSetKSP_KSP);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
